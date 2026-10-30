"""Wires the 4 components into one run: Ingestion -> Validation ->
Transformation -> Trainer, then optionally syncs artifacts/model to S3.
`python main.py` just calls `TrainingPipeline().run_pipeline()` — this is
the single entry point lecture 296 ("Model Training Pipeline Implementation")
builds toward, and the same thing a GitHub Actions CD job or a scheduled
retraining job would call.
"""
import os
import sys

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.constants.training_pipeline import SAVED_MODEL_DIR, TRAINING_BUCKET_NAME
from networksecurity.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelTrainerConfig,
    TrainingPipelineConfig,
)
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


class TrainingPipeline:
    def __init__(self, mongo_db_url: str | None = None):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.mongo_db_url = mongo_db_url or os.getenv("MONGO_DB_URL")

    def start_data_ingestion(self):
        config = DataIngestionConfig(self.training_pipeline_config)
        ingestion = DataIngestion(config, mongo_db_url=self.mongo_db_url)
        logging.info("Starting data ingestion")
        return ingestion.initiate_data_ingestion()

    def start_data_validation(self, data_ingestion_artifact):
        config = DataValidationConfig(self.training_pipeline_config)
        validation = DataValidation(data_ingestion_artifact, config)
        logging.info("Starting data validation")
        return validation.initiate_data_validation()

    def start_data_transformation(self, data_validation_artifact):
        config = DataTransformationConfig(self.training_pipeline_config)
        transformation = DataTransformation(data_validation_artifact, config)
        logging.info("Starting data transformation")
        return transformation.initiate_data_transformation()

    def start_model_trainer(self, data_transformation_artifact):
        config = ModelTrainerConfig(self.training_pipeline_config)
        trainer = ModelTrainer(config, data_transformation_artifact)
        logging.info("Starting model training")
        return trainer.initiate_model_trainer()

    def sync_artifact_dir_to_s3(self):
        """Best-effort: only runs if the `aws` CLI + credentials are set up.
        Skipped (with a warning, not a crash) if AWS isn't configured yet —
        training still succeeds locally either way."""
        try:
            from networksecurity.cloud.s3_syncer import S3Sync

            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            S3Sync.sync_folder_to_s3(self.training_pipeline_config.artifact_dir, aws_bucket_url)
            S3Sync.sync_folder_to_s3(SAVED_MODEL_DIR, f"s3://{TRAINING_BUCKET_NAME}/final_model")
        except Exception as e:
            logging.warning(f"Skipping S3 sync (AWS not configured?): {e}")

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            if not data_validation_artifact.validation_status:
                logging.warning("Data validation found drift/errors; continuing anyway for this training run.")
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            self.sync_artifact_dir_to_s3()
            logging.info(f"Pipeline complete. Model at {model_trainer_artifact.trained_model_file_path}")
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
