"""Batch Prediction: run the already-trained, saved model over a CSV of new
records in one shot (as opposed to the FastAPI app, which predicts one
request at a time). Useful for scoring a whole backlog of URLs/network
records at once rather than one API call per row.
"""
import os
import sys

import pandas as pd

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import load_object


def run_batch_prediction(input_csv_path: str, output_csv_path: str, model_path: str = os.path.join("saved_models", "model.pkl")) -> pd.DataFrame:
    try:
        logging.info(f"Running batch prediction on {input_csv_path}")
        network_model = load_object(model_path)

        df = pd.read_csv(input_csv_path)
        predictions = network_model.predict(df)

        df["predicted_column"] = predictions
        os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        logging.info(f"Batch predictions written to {output_csv_path}")
        return df
    except Exception as e:
        raise NetworkSecurityException(e, sys)
