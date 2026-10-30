"""Evaluate a saved AAPL forecasting model on a prepared feature table."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from formative_1.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN


def evaluate(
    feature_parquet: str = "data/processed/aapl_features.parquet",
    model_file: str = "models/aapl_close_model.joblib",
):
    data = pl.read_parquet(feature_parquet)
    model = joblib.load(model_file)
    X = data.select(FEATURE_COLUMNS).to_numpy()
    y = data[TARGET_COLUMN].to_numpy()
    predictions = model.predict(X)
    metrics = {
        "MAE": float(mean_absolute_error(y, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y, predictions))),
        "R2": float(r2_score(y, predictions)),
    }
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    evaluate()
