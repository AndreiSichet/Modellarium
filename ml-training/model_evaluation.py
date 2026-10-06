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

from calibration import brier_score, expected_calibration_error
from common import FEATURE_COLUMNS, MONEYLINE_TARGET
from train_baseline import REGRESSION_TARGETS

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

def classification_metrics(y_true, proba, labels) -> dict:
    """The one definition of the probability metrics, for every caller."""
    # float64 BEFORE scoring, not after. XGBoost returns float32, and summing
    # a couple of thousand log terms at that width lands ~4e-08 from the
    # float64 answer - invisible at four decimals, which is why this module
    # and the training scripts quietly disagreed in the eighth decimal until
    # they were brought together. It also keeps a flat segment from wobbling
    # by an ULP across a calibration bin edge. See measure_calibration.
    proba = np.asarray(proba, dtype=np.float64)
    return {
        "log_loss": float(log_loss(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, labels)),
        "brier": brier_score(y_true, proba),
        "ece": expected_calibration_error(y_true, proba),
    }

def regression_metrics(y_true, predictions) -> dict:
    """The one definition of the regression metrics, for every caller."""
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
    }

def evaluate(model, frame, target: str, classification: bool) -> dict:
    """Score one model on one dataframe. Never trains, never mutates."""
    x = frame[FEATURE_COLUMNS]
    y = frame[target]

    if classification:
        return classification_metrics(y, model.predict_proba(x)[:, 1], model.predict(x))

    return regression_metrics(y, model.predict(x))

def primary(metrics: dict, classification: bool) -> float:
    """The one number the gate compares. See the module docstring."""
    return metrics[PRIMARY_METRIC[classification]]
