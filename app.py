"""FastAPI service: /train kicks off the full training pipeline, /predict
scores an uploaded CSV of network/URL records using the currently saved
model. This is the "Backend, Cloud & MLOps: FastAPI (REST APIs)" piece of
the resume for this project.
"""
import os
import sys

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object

app = FastAPI(title="Network Security - Phishing Detection")

MODEL_PATH = os.path.join("saved_models", "model.pkl")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/train")
def train():
    try:
        pipeline = TrainingPipeline()
        artifact = pipeline.run_pipeline()
        return {
            "message": "Training completed",
            "model_path": artifact.trained_model_file_path,
            "test_f1_score": artifact.test_metric_artifact.f1_score,
        }
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Upload a CSV with the same 30 feature columns as training data (no
    `result` column needed) and get back phishing/legitimate predictions."""
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="No trained model found yet. Call /train first.")
    try:
        df = pd.read_csv(file.file)
        network_model = load_object(MODEL_PATH)
        predictions = network_model.predict(df)
        # Training remaps the original {-1, 1} labels to {0, 1}: 1 stays "legitimate", -1 becomes 0 ("phishing")
        df["prediction"] = ["legitimate" if p == 1 else "phishing" for p in predictions]
        return JSONResponse(content=df.to_dict(orient="records"))
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
