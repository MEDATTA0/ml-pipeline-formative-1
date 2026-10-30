"""Leakage-aware features for next-trading-session AAPL close forecasting."""

from __future__ import annotations

import polars as pl

FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits",
    "Daily_Return", "Close_Lag_1", "Close_Lag_2", "Close_Lag_5",
    "Close_Lag_10", "MA_7", "MA_30",
]
TARGET_COLUMN = "Target_Close"


def make_features(df: pl.DataFrame, require_target: bool = True) -> pl.DataFrame:
    """Create features from information available at session t.

    The target and target date are shifted one row forward, so t+1 means the
    next available trading observation, not the next calendar date.
    Rolling means include the current close, which is known at prediction time.
    With require_target=False the final session is kept (its target is null) so
    it can be used for inference.
    """
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing columns needed for features: {missing}")

    expressions = []
    for col in ["Dividends", "Stock Splits"]:
        if col not in df.columns:
            expressions.append(pl.lit(0.0).alias(col))
    if expressions:
        df = df.with_columns(expressions)

    featured = (
        df.sort("Date")
        .with_columns(
            (pl.col("Close") / pl.col("Close").shift(1) - 1).alias("Daily_Return"),
            pl.col("Close").shift(1).alias("Close_Lag_1"),
            pl.col("Close").shift(2).alias("Close_Lag_2"),
            pl.col("Close").shift(5).alias("Close_Lag_5"),
            pl.col("Close").shift(10).alias("Close_Lag_10"),
            pl.col("Close").rolling_mean(window_size=7, min_samples=7).alias("MA_7"),
            pl.col("Close").rolling_mean(window_size=30, min_samples=30).alias("MA_30"),
            pl.col("Close").shift(-1).alias(TARGET_COLUMN),
            pl.col("Date").shift(-1).alias("Target_Date"),
        )
    )

    return featured.select(
        "Date", "Target_Date", *FEATURE_COLUMNS, TARGET_COLUMN
    ).drop_nulls(
        subset=[*FEATURE_COLUMNS, *([TARGET_COLUMN] if require_target else [])]
    )
