"""Load a saved model and score it against a given dataframe."""

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)
from xgboost import XGBClassifier, XGBRegressor

from common import FEATURE_COLUMNS
from train_baseline import REGRESSION_TARGETS
from train_moneyline_xgb import TARGET as MONEYLINE_TARGET

MODELS_DIR = Path(__file__).resolve().parent / "models"

TASKS = [("moneyline", "Moneyline", MONEYLINE_TARGET, True)] + [
    (label.lower().replace(" ", "_"), label, target, False)
    for target, label, _stat, _combine in REGRESSION_TARGETS
]

PRIMARY_METRIC = {True: "log_loss", False: "mae"}

def load_model(key: str, classification: bool, directory: Path = None):
    """One saved model, as the sklearn-API estimator that wrote it."""
    directory = directory or MODELS_DIR
    path = directory / f"{key}.json"
    if not path.exists():
        raise FileNotFoundError(f"no model at {path}")

    model = XGBClassifier() if classification else XGBRegressor()
    model.load_model(str(path))
    return model

def evaluate(model, frame, target: str, classification: bool) -> dict:
    """Score one model on one dataframe. Never trains, never mutates."""
    x = frame[FEATURE_COLUMNS]
    y = frame[target]

    if classification:
        proba = model.predict_proba(x)[:, 1]
        return {
            "log_loss": float(log_loss(y, proba)),
            "accuracy": float(accuracy_score(y, model.predict(x))),
        }

    predictions = model.predict(x)
    return {
        "mae": float(mean_absolute_error(y, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
    }

def primary(metrics: dict, classification: bool) -> float:
    """The one number the gate compares. See the module docstring."""
    return metrics[PRIMARY_METRIC[classification]]
