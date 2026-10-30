"""ETL "Extract + Load" step: read the raw phishing dataset CSV and load it
into MongoDB as the single source of truth the training pipeline ingests
from. Run once (or whenever the source CSV changes) — not part of every
training run, which is why it's a standalone script rather than a pipeline
component.

MONGO_DB_URL can point at a local MongoDB (`mongodb://localhost:27017`) or a
MongoDB Atlas cluster — identical code either way, only the .env value
changes.
"""
import json
import os
import sys

import certifi
import pandas as pd
import pymongo
from dotenv import load_dotenv

from networksecurity.constants.training_pipeline import DATA_INGESTION_COLLECTION_NAME, DATA_INGESTION_DATABASE_NAME
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL", "mongodb://localhost:27017")
ca = certifi.where()  # TLS root certs, only actually needed for Atlas (mongodb+srv://) connections


class NetworkDataExtract:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def csv_to_json_converter(self, file_path: str) -> list[dict]:
        try:
            data = pd.read_csv(file_path)
            data = data.reset_index(drop=True)
            return json.loads(data.T.to_json()).values()
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def insert_data_mongodb(self, records, database: str, collection: str) -> int:
        try:
            records = list(records)
            use_tls = MONGO_DB_URL.startswith("mongodb+srv://")
            client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca if use_tls else None)
            db = client[database]
            db[collection].delete_many({})  # idempotent: re-running this script doesn't duplicate rows
            db[collection].insert_many(records)
            return len(records)
        except Exception as e:
            raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    FILE_PATH = os.path.join("Network_Data", "phisingData.csv")
    extractor = NetworkDataExtract()
    records = extractor.csv_to_json_converter(FILE_PATH)
    count = extractor.insert_data_mongodb(records, DATA_INGESTION_DATABASE_NAME, DATA_INGESTION_COLLECTION_NAME)
    logging.info(f"Inserted {count} records into {DATA_INGESTION_DATABASE_NAME}.{DATA_INGESTION_COLLECTION_NAME}")
    print(f"Inserted {count} records into MongoDB")
