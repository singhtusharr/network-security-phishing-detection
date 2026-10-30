"""Config objects: each pipeline stage gets a dataclass with the exact
filesystem paths it needs, all derived from one timestamped run directory.
This is what makes every training run reproducible and non-destructive —
each run writes into Artifacts/<timestamp>/... instead of overwriting the
previous run's outputs.
"""
import os
from dataclasses import dataclass, field
from datetime import datetime

from networksecurity.constants import training_pipeline as tp


@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = tp.PIPELINE_NAME
    artifact_name: str = tp.ARTIFACT_DIR
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
    artifact_dir: str = field(init=False)
    model_dir: str = field(init=False)

    def __post_init__(self):
        self.artifact_dir = os.path.join(self.artifact_name, self.timestamp)
        self.model_dir = os.path.join(tp.SAVED_MODEL_DIR)


@dataclass
class DataIngestionConfig:
    training_pipeline_config: TrainingPipelineConfig
    data_ingestion_dir: str = field(init=False)
    feature_store_file_path: str = field(init=False)
    training_file_path: str = field(init=False)
    testing_file_path: str = field(init=False)
    train_test_split_ratio: float = tp.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
    collection_name: str = tp.DATA_INGESTION_COLLECTION_NAME
    database_name: str = tp.DATA_INGESTION_DATABASE_NAME

    def __post_init__(self):
        base = os.path.join(self.training_pipeline_config.artifact_dir, tp.DATA_INGESTION_DIR_NAME)
        self.data_ingestion_dir = base
        self.feature_store_file_path = os.path.join(base, tp.DATA_INGESTION_FEATURE_STORE_DIR, tp.FILE_NAME)
        self.training_file_path = os.path.join(base, tp.DATA_INGESTION_INGESTED_DIR, tp.TRAIN_FILE_NAME)
        self.testing_file_path = os.path.join(base, tp.DATA_INGESTION_INGESTED_DIR, tp.TEST_FILE_NAME)


@dataclass
class DataValidationConfig:
    training_pipeline_config: TrainingPipelineConfig
    valid_train_file_path: str = field(init=False)
    valid_test_file_path: str = field(init=False)
    invalid_train_file_path: str = field(init=False)
    invalid_test_file_path: str = field(init=False)
    drift_report_file_path: str = field(init=False)

    def __post_init__(self):
        base = os.path.join(self.training_pipeline_config.artifact_dir, tp.DATA_VALIDATION_DIR_NAME)
        self.valid_train_file_path = os.path.join(base, tp.DATA_VALIDATION_VALID_DIR, tp.TRAIN_FILE_NAME)
        self.valid_test_file_path = os.path.join(base, tp.DATA_VALIDATION_VALID_DIR, tp.TEST_FILE_NAME)
        self.invalid_train_file_path = os.path.join(base, tp.DATA_VALIDATION_INVALID_DIR, tp.TRAIN_FILE_NAME)
        self.invalid_test_file_path = os.path.join(base, tp.DATA_VALIDATION_INVALID_DIR, tp.TEST_FILE_NAME)
        self.drift_report_file_path = os.path.join(
            base, tp.DATA_VALIDATION_DRIFT_REPORT_DIR, tp.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
        )


@dataclass
class DataTransformationConfig:
    training_pipeline_config: TrainingPipelineConfig
    transformed_train_file_path: str = field(init=False)
    transformed_test_file_path: str = field(init=False)
    transformed_object_file_path: str = field(init=False)

    def __post_init__(self):
        base = os.path.join(self.training_pipeline_config.artifact_dir, tp.DATA_TRANSFORMATION_DIR_NAME)
        self.transformed_train_file_path = os.path.join(
            base, tp.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, tp.TRAIN_FILE_NAME.replace("csv", "npy")
        )
        self.transformed_test_file_path = os.path.join(
            base, tp.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, tp.TEST_FILE_NAME.replace("csv", "npy")
        )
        self.transformed_object_file_path = os.path.join(
            base, tp.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR, tp.PREPROCESSING_OBJECT_FILE_NAME
        )


@dataclass
class ModelTrainerConfig:
    training_pipeline_config: TrainingPipelineConfig
    trained_model_file_path: str = field(init=False)
    expected_accuracy: float = tp.MODEL_TRAINER_EXPECTED_SCORE
    overfitting_underfitting_threshold: float = tp.MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD

    def __post_init__(self):
        base = os.path.join(self.training_pipeline_config.artifact_dir, tp.MODEL_TRAINER_DIR_NAME)
        self.trained_model_file_path = os.path.join(
            base, tp.MODEL_TRAINER_TRAINED_MODEL_DIR, tp.MODEL_TRAINER_TRAINED_MODEL_NAME
        )
