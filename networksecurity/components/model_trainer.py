"""Model Trainer: trains several candidate algorithms with hyperparameter
tuning (GridSearchCV, in evaluate_models), picks whichever scores best on the
held-out test set, logs every run's params/metrics/artifact to MLflow, and
saves the winning model bundled with its preprocessor.

MLflow tracking URI: if MLFLOW_TRACKING_URI is set in the environment (e.g.
pointing at a DagsHub-hosted remote), runs are logged there; otherwise MLflow
falls back to local file-based tracking at ./mlruns automatically — same
code either way, which is exactly the point of using MLflow instead of
printing metrics to a notebook cell.
"""
import os
import sys

import mlflow
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from networksecurity.entity.artifact_entity import (
    ClassificationMetricArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from networksecurity.constants.training_pipeline import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_PRODUCTION_ALIAS,
    MLFLOW_REGISTERED_MODEL_NAME,
)
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import (
    evaluate_models,
    get_classification_score,
    load_numpy_array_data,
    load_object,
    save_object,
)
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

# random_state=42 on every model that uses randomness, so a re-run gives the same scores.
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
}

PARAM_GRIDS = {
    "Logistic Regression": {"C": [0.1, 1.0, 10.0]},
    "Decision Tree": {"criterion": ["gini", "entropy"], "max_depth": [5, 10, None]},
    "Random Forest": {"n_estimators": [50, 100], "max_depth": [5, 10, None]},
    "Gradient Boosting": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
    "AdaBoost": {"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]},
    "XGBoost": {"n_estimators": [50, 100], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
}


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            self._init_mlflow()
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def _init_mlflow():
        dagshub_owner = os.getenv("DAGSHUB_REPO_OWNER")
        dagshub_repo = os.getenv("DAGSHUB_REPO_NAME")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if dagshub_owner and dagshub_repo:
            # Hosted remote tracking + model registry on DagsHub. Needs a DagsHub
            # account; dagshub.init points MLflow at that repo's MLflow server.
            import dagshub

            dagshub.init(repo_owner=dagshub_owner, repo_name=dagshub_repo, mlflow=True)
        elif tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    @staticmethod
    def track_mlflow(run_name: str, model, metric: ClassificationMetricArtifact, register: bool = False):
        with mlflow.start_run(run_name=run_name):
            mlflow.log_metric("f1_score", metric.f1_score)
            mlflow.log_metric("precision", metric.precision_score)
            mlflow.log_metric("recall", metric.recall_score)
            # pickle, not the newer skops default: skops refuses to deserialize
            # sklearn's internal Tree type for safety, but this is a model we
            # just trained ourselves, not an untrusted file from elsewhere.
            mlflow.sklearn.log_model(
                model,
                "model",
                serialization_format="pickle",
                registered_model_name=MLFLOW_REGISTERED_MODEL_NAME if register else None,
            )
        if register:
            ModelTrainer._promote_latest_version()

    @staticmethod
    def _promote_latest_version():
        """Point the 'champion' alias at the newest registered version (MLflow's
        replacement for the old Staging/Production stages)."""
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{MLFLOW_REGISTERED_MODEL_NAME}'")
        latest = max(versions, key=lambda v: int(v.version))
        client.set_registered_model_alias(MLFLOW_REGISTERED_MODEL_NAME, MLFLOW_PRODUCTION_ALIAS, latest.version)
        logging.info(f"Model version {latest.version} registered; alias '{MLFLOW_PRODUCTION_ALIAS}' now points to it")

    def train_model(self, X_train, y_train, X_test, y_test) -> ModelTrainerArtifact:
        try:
            report = evaluate_models(X_train, y_train, X_test, y_test, MODELS, PARAM_GRIDS)
            best_model_name = max(report, key=report.get)
            best_model = MODELS[best_model_name]
            logging.info(f"Best model: {best_model_name} (test f1={report[best_model_name]:.4f})")

            y_train_pred = best_model.predict(X_train)
            train_metric = ClassificationMetricArtifact(**get_classification_score(y_train, y_train_pred))

            y_test_pred = best_model.predict(X_test)
            test_metric = ClassificationMetricArtifact(**get_classification_score(y_test, y_test_pred))

            self.track_mlflow(f"{best_model_name}-train", best_model, train_metric)
            # Only a model that clears the F1 gate is registered (and becomes 'champion').
            passed_gate = test_metric.f1_score >= self.model_trainer_config.expected_accuracy
            self.track_mlflow(f"{best_model_name}-test", best_model, test_metric, register=passed_gate)

            if test_metric.f1_score < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"No model met the expected accuracy of {self.model_trainer_config.expected_accuracy}; "
                    f"best was {best_model_name} at {test_metric.f1_score:.4f}"
                )

            overfit_gap = train_metric.f1_score - test_metric.f1_score
            if overfit_gap > self.model_trainer_config.overfitting_underfitting_threshold:
                logging.warning(f"Possible overfitting: train/test f1 gap = {overfit_gap:.4f}")

            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)
            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, network_model)
            save_object(os.path.join("saved_models", "model.pkl"), network_model)

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric,
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            return self.train_model(X_train, y_train, X_test, y_test)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
