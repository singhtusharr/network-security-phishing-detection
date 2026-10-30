# Network Security System — Phishing Detection Pipeline

End-to-end MLOps pipeline that classifies whether a website is phishing or
legitimate, from 30 URL/page-structure features (UCI "Phishing Websites"
dataset, 11,055 records).

## Architecture

```
phisingData.csv
    │  push_data.py (ETL: Extract + Load)
    ▼
MongoDB  ──────────────────────────────────────────────────────────┐
    │  DataIngestion (Extract + Transform: train/test split)       │
    ▼                                                               │
Artifacts/<run>/data_ingestion/ingested/{train,test}.csv            │
    │  DataValidation (schema check + KS-test drift detection)     │
    ▼                                                               │
Artifacts/<run>/data_validation/validated/{train,test}.csv          │
    │  DataTransformation (KNNImputer -> numpy arrays)              │
    ▼                                                               │
Artifacts/<run>/data_transformation/transformed/{train,test}.npy    │
    │  ModelTrainer (5 algorithms x GridSearchCV, pick best by F1) │
    ▼                                                               │
saved_models/model.pkl (preprocessor + model bundled)                │
    │  logged to MLflow (local or DagsHub remote)                  │
    ▼                                                               │
S3 (optional, if AWS configured) ──► Docker ──► GitHub Actions ──► EC2
```

## Why each piece is there

- **MongoDB as the ETL target**: the raw CSV is loaded once into Mongo
  (`push_data.py`); the pipeline always ingests from Mongo, not the CSV
  directly, so swapping in a different/updated data source later only means
  changing what gets pushed to Mongo, not touching the pipeline code.
- **Data Validation / drift detection**: a 2-sample Kolmogorov-Smirnov test
  per column between train and test catches distribution mismatches that
  would make test-set metrics misleading.
- **KNNImputer over SimpleImputer**: these phishing-indicator features are
  correlated with each other, so a similar row's values are a better guess
  for a missing value than the column mean.
- **5 candidate algorithms with GridSearchCV**: Logistic Regression, Decision
  Tree, Random Forest, Gradient Boosting, AdaBoost, and XGBoost are each
  hyperparameter-tuned, then compared on held-out test F1 — "Model Trainer
  And Evaluation With Hyperparameter Tuning" means a real per-algorithm grid
  search and comparison, not one `.fit()` call.
- **MLflow experiment tracking**: every training run's params/metrics/model
  are logged, so you can compare runs instead of relying on memory/print
  statements. Works identically against local file storage or a DagsHub
  remote — only `MLFLOW_TRACKING_URI` changes.
- **NetworkModel wrapper**: bundles the fitted preprocessor + classifier into
  one pickled object, so inference code can never apply the model to
  un-preprocessed input or mismatch a preprocessor/model pair.
- **S3 sync / Docker / GitHub Actions / EC2**: the Model Pusher syncs
  artifacts to S3, GitHub Actions builds+pushes a Docker image to ECR on
  every push to `main`, and a self-hosted runner on the EC2 box pulls and
  restarts the container — full CI/CD, matching the resume's deployment
  claim.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`.env`:
```
MONGO_DB_URL=mongodb://localhost:27017      # or a MongoDB Atlas connection string
MLFLOW_TRACKING_URI=                         # leave blank for local ./mlruns, or set to a DagsHub remote
```

Local MongoDB via Docker (no Atlas account needed):
```bash
docker run -d --name mongo-local -p 27017:27017 mongo:7
```

## Run

```bash
# 1. Load the dataset into MongoDB (once)
python push_data.py

# 2. Run the full training pipeline
python main.py

# 3. Serve predictions
uvicorn app:app --reload
# GET  /train    -> retrain from scratch
# POST /predict  -> upload a CSV, get phishing/legitimate predictions
```

## Study notebook

`notebook/network_security_eda.ipynb` walks through EDA and the same
pipeline stages interactively.
