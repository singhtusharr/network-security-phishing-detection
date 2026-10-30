"""Fast unit tests: no MongoDB, AWS or network needed, so they also run in CI."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.data_validation import DataValidation
from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    TrainingPipelineConfig,
)
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.utils.main_utils.utils import (
    get_classification_score,
    load_numpy_array_data,
    load_object,
    read_yaml_file,
    save_numpy_array_data,
    save_object,
    write_yaml_file,
)
from networksecurity.utils.ml_utils.model.estimator import NetworkModel


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A scratch project folder with a 3-column schema.yaml; the code reads it by relative path."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data_schema").mkdir()
    (tmp_path / "data_schema" / "schema.yaml").write_text(
        "columns:\n- f1: float64\n- f2: float64\n- result: int64\n"
    )
    return tmp_path


def _frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n), "result": rng.choice([-1, 1], size=n)})


def test_exception_reports_file_line_and_original_message():
    try:
        try:
            1 / 0
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    except NetworkSecurityException as err:
        text = str(err)
    assert "Error occurred in python script" in text
    assert "test_pipeline_units.py" in text
    assert "division by zero" in text


def test_save_and_load_object_roundtrip(tmp_path):
    path = str(tmp_path / "nested" / "obj.pkl")
    save_object(path, {"a": 1, "b": [1, 2, 3]})
    assert load_object(path) == {"a": 1, "b": [1, 2, 3]}


def test_load_object_missing_file_raises(tmp_path):
    with pytest.raises(NetworkSecurityException):
        load_object(str(tmp_path / "missing.pkl"))


def test_yaml_roundtrip_and_replace(tmp_path):
    path = str(tmp_path / "r" / "report.yaml")
    write_yaml_file(path, {"x": {"p_value": 0.5}})
    write_yaml_file(path, {"y": 1}, replace=True)
    assert read_yaml_file(path) == {"y": 1}


def test_numpy_roundtrip(tmp_path):
    path = str(tmp_path / "arr" / "a.npy")
    array = np.arange(12).reshape(3, 4)
    save_numpy_array_data(path, array)
    assert np.array_equal(load_numpy_array_data(path), array)


def test_classification_scores_known_values():
    perfect = get_classification_score([1, 0, 1, 0], [1, 0, 1, 0])
    assert perfect == {"f1_score": 1.0, "precision_score": 1.0, "recall_score": 1.0}
    partial = get_classification_score([1, 1, 0, 0], [1, 0, 0, 0])  # precision 1.0, recall 0.5
    assert partial["precision_score"] == 1.0
    assert partial["recall_score"] == 0.5
    assert partial["f1_score"] == pytest.approx(2 * 1.0 * 0.5 / 1.5)


def test_network_model_imputes_before_predicting():
    train = pd.DataFrame({"a": [0.0, 1.0, 0.0, 1.0, 1.0, 0.0], "b": [1.0, 0.0, 1.0, 0.0, 0.0, 1.0]})
    labels = [1, 0, 1, 0, 0, 1]
    preprocessor = Pipeline([("imputer", KNNImputer(n_neighbors=2))]).fit(train)
    model = LogisticRegression().fit(preprocessor.transform(train), labels)
    network_model = NetworkModel(preprocessor=preprocessor, model=model)
    new_rows = pd.DataFrame({"a": [np.nan, 1.0], "b": [1.0, np.nan]})  # contains NaNs on purpose
    predictions = network_model.predict(new_rows)
    assert len(predictions) == 2
    assert set(predictions) <= {0, 1}


def test_all_configs_share_one_timestamped_run_folder():
    root = TrainingPipelineConfig()
    ingestion = DataIngestionConfig(root)
    validation = DataValidationConfig(root)
    assert ingestion.training_file_path.split(os.sep)[0] == "Artifacts"
    assert root.timestamp in ingestion.training_file_path
    assert root.timestamp in validation.valid_train_file_path
    assert ingestion.training_file_path.endswith(os.path.join("ingested", "train.csv"))


def test_validation_checks_column_count_against_schema(workspace):
    base = _frame()
    validation = DataValidation(DataIngestionArtifact("unused", "unused"), DataValidationConfig(TrainingPipelineConfig()))
    assert validation.validate_number_of_columns(base) is True
    assert validation.validate_number_of_columns(base.assign(extra=1)) is False


def test_drift_detection_flags_shifted_distribution(workspace):
    base = _frame()
    validation = DataValidation(DataIngestionArtifact("unused", "unused"), DataValidationConfig(TrainingPipelineConfig()))
    assert validation.detect_dataset_drift(base, base.copy()) is True  # identical samples: p-value 1.0
    shifted = base.copy()
    shifted["f1"] = shifted["f1"] + 5.0
    assert validation.detect_dataset_drift(base, shifted) is False
    report = read_yaml_file(validation.data_validation_config.drift_report_file_path)
    assert report["f1"]["drift_status"] is True


def test_transformation_imputes_remaps_labels_and_saves_artifacts(workspace):
    train, test = _frame(seed=1), _frame(n=100, seed=2)
    train.loc[0, "f1"] = np.nan  # a missing value the imputer must fill
    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)
    validation_artifact = DataValidationArtifact(True, "train.csv", "test.csv", "x", "y", "z")
    config = DataTransformationConfig(TrainingPipelineConfig())
    artifact = DataTransformation(validation_artifact, config).initiate_data_transformation()

    train_arr = load_numpy_array_data(artifact.transformed_train_file_path)
    test_arr = load_numpy_array_data(artifact.transformed_test_file_path)
    assert train_arr.shape == (300, 3) and test_arr.shape == (100, 3)  # 2 features + label
    assert not np.isnan(train_arr).any()
    assert set(np.unique(train_arr[:, -1])) <= {0.0, 1.0}  # -1 was remapped to 0
    assert load_object(artifact.transformed_object_file_path) is not None
