"""Every magic string/number the pipeline uses, in one place, so changing
e.g. the train/test split ratio or an artifact filename doesn't mean hunting
through 6 component files.
"""
import os

TARGET_COLUMN = "result"
PIPELINE_NAME = "NetworkSecurity"
ARTIFACT_DIR = "Artifacts"
FILE_NAME = "phisingData.csv"

TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")
SAVED_MODEL_DIR = os.path.join("saved_models")
MODEL_FILE_NAME = "model.pkl"

# --- Data Ingestion ---
DATA_INGESTION_COLLECTION_NAME = "NetworkData"
DATA_INGESTION_DATABASE_NAME = "NETWORK_SECURITY"
DATA_INGESTION_DIR_NAME = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR = "feature_store"
DATA_INGESTION_INGESTED_DIR = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION = 0.2

# --- Data Validation ---
DATA_VALIDATION_DIR_NAME = "data_validation"
DATA_VALIDATION_VALID_DIR = "validated"
DATA_VALIDATION_INVALID_DIR = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME = "report.yaml"

# --- Data Transformation ---
DATA_TRANSFORMATION_DIR_NAME = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR = "transformed_object"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
# KNNImputer params: features are highly correlated (all binary/ternary
# phishing indicators), so nearest-neighbor imputation fills gaps using
# similar rows rather than a blind column mean.
DATA_TRANSFORMATION_IMPUTER_PARAMS = {
    "missing_values": float("nan"),
    "n_neighbors": 3,
    "weights": "uniform",
}

# --- Model Trainer ---
MODEL_TRAINER_DIR_NAME = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE = 0.6
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD = 0.05

# --- Model Pusher ---
SAVED_MODEL_MODEL_DIR = os.path.join(SAVED_MODEL_DIR, PIPELINE_NAME)

# --- MLflow ---
MLFLOW_EXPERIMENT_NAME = "NetworkSecurityPhishingDetection"

# --- AWS S3 (model pusher's final artifact sync) ---
TRAINING_BUCKET_NAME = "networksecurity-model-artifacts"
