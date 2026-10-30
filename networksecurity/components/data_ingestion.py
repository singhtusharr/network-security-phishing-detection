"""Data Ingestion: pull the raw records out of MongoDB (where `push_data.py`
loaded them as the ETL "Load" step), dump a feature-store CSV snapshot of the
full dataset, then split into train/test — this is the first stage of the
training pipeline and produces the DataIngestionArtifact every later stage
builds on.
"""
import os
import sys

import numpy as np
import pandas as pd
import pymongo
from sklearn.model_selection import train_test_split

from networksecurity.entity.artifact_entity import DataIngestionArtifact
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig, mongo_db_url: str):
        try:
            self.data_ingestion_config = data_ingestion_config
            self.mongo_db_url = mongo_db_url
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """Pull every document from the Mongo collection into a DataFrame.
        Mongo auto-adds an `_id` field we don't want as a model feature, and
        the source data uses "na" as its missing-value sentinel string."""
        try:
            client = pymongo.MongoClient(self.mongo_db_url)
            collection = client[self.data_ingestion_config.database_name][self.data_ingestion_config.collection_name]
            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns:
                df = df.drop(columns=["_id"])
            df.replace({"na": np.nan}, inplace=True)
            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            feature_store_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_path), exist_ok=True)
            dataframe.to_csv(feature_store_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio, random_state=42
            )
            logging.info(f"Train/test split done: {len(train_set)} train rows, {len(test_set)} test rows")

            os.makedirs(os.path.dirname(self.data_ingestion_config.training_file_path), exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            return DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)
