"""Small, shared I/O helpers used across components — reading/writing yaml,
numpy arrays, and pickled objects (preprocessor, trained model). Centralizing
these means every component serializes artifacts the exact same way.
"""
import os
import pickle
import sys

import numpy as np
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            yaml.dump(content, f)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            np.save(f, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def load_numpy_array_data(file_path: str) -> np.ndarray:
    try:
        with open(file_path, "rb") as f:
            return np.load(f)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def save_object(file_path: str, obj: object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logging.info(f"Saved object to {file_path}")
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def get_classification_score(y_true, y_pred) -> dict:
    """Returns f1/precision/recall as a dict ready to drop into a
    ClassificationMetricArtifact or log straight to MLflow."""
    return {
        "f1_score": f1_score(y_true, y_pred),
        "precision_score": precision_score(y_true, y_pred),
        "recall_score": recall_score(y_true, y_pred),
    }


def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict) -> dict:
    """Hyperparameter-tune each candidate model with GridSearchCV (cv=3),
    refit on the best params, and score every model on the held-out test
    set. Returns {model_name: test_f1_score} so the caller can pick the
    best performer — this is what "Model Trainer And Evaluation With
    Hyperparameter Tuning" is: not just .fit(), but a grid search per
    algorithm plus an apples-to-apples comparison across algorithms.
    """
    try:
        report = {}
        for model_name, model in models.items():
            param_grid = params.get(model_name, {})
            gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            test_f1 = f1_score(y_test, y_test_pred)
            report[model_name] = test_f1
            logging.info(f"{model_name}: best_params={gs.best_params_}, test_f1={test_f1:.4f}")
        return report
    except Exception as e:
        raise NetworkSecurityException(e, sys)
