"""Data Validation: gate-keeps the pipeline against bad data BEFORE it ever
reaches training. Two checks:

1. Schema check — does the incoming data have the expected number of columns
   (data_schema/schema.yaml)? Catches a broken ETL run / malformed export
   early, rather than letting a model silently train on the wrong features.
2. Dataset drift check — per numeric column, a two-sample Kolmogorov-Smirnov
   test comparing the train and test distributions. A low p-value (<0.05)
   means train and test look like they come from different distributions,
   which would make test-set metrics a misleading estimate of real-world
   performance.
"""
import os
import sys

import pandas as pd
from scipy.stats import ks_2samp

from networksecurity.constants.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        expected_columns = len(self._schema_config["columns"])
        actual_columns = len(dataframe.columns)
        logging.info(f"Required columns: {expected_columns}, dataframe columns: {actual_columns}")
        return actual_columns == expected_columns

    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> bool:
        try:
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                result = ks_2samp(d1, d2)
                drift_found = result.pvalue < threshold
                if drift_found:
                    status = False
                report[column] = {"p_value": float(result.pvalue), "drift_status": bool(drift_found)}

            write_yaml_file(self.data_validation_config.drift_report_file_path, content=report, replace=True)
            return status
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            errors = []
            if not self.validate_number_of_columns(train_df):
                errors.append("Train dataframe does not contain the expected number of columns")
            if not self.validate_number_of_columns(test_df):
                errors.append("Test dataframe does not contain the expected number of columns")

            drift_status = self.detect_dataset_drift(train_df, test_df)
            validation_status = drift_status and not errors

            if errors:
                logging.warning(f"Data validation errors: {errors}")

            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)
            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            return DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)
