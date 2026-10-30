"""Train and compare chronological models for next-session AAPL closing price."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from formative_1.feature_engineering import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    make_features,
)
from formative_1.preprocessing import load_source, validate_and_clean


RANDOM_STATE = 42

RF_PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    """Calculate regression evaluation metrics."""

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def tune_random_forest(X_train, y_train):
    """Tune a Random Forest using chronological cross-validation."""

    cv = TimeSeriesSplit(n_splits=5)

    search = GridSearchCV(
        estimator=RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        param_grid=RF_PARAM_GRID,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    search.fit(X_train, y_train)

    return search


def save_prediction_plot(
    dates,
    actual,
    predictions,
    title,
    output_path,
):
    """Save an actual-versus-predicted time-series plot."""

    plt.figure(figsize=(13, 6))

    plt.plot(
        dates,
        actual,
        label="Actual",
        linewidth=1.5,
    )

    plt.plot(
        dates,
        predictions,
        label="Predicted",
        linewidth=1.2,
    )

    plt.title(title)
    plt.xlabel("Target trading date")
    plt.ylabel("Closing price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

def walk_forward_evaluation(
    featured: pl.DataFrame,
    return_params: dict,
    output_dir: str = "outputs",
    n_splits: int = 5,
    test_size: int = 250,
):
    """
    Evaluate forecasting models using expanding-window
    walk-forward validation.

    Hyperparameters are selected using the initial training
    period only, then kept fixed across the evaluation folds.
    """

    X = featured.select(FEATURE_COLUMNS).to_numpy()
    y = featured[TARGET_COLUMN].to_numpy()
    current_close = featured["Close"].to_numpy()
    dates = featured["Target_Date"].to_list()

    if featured.height <= n_splits * test_size:
        raise ValueError(
            "Not enough observations for the requested "
            "walk-forward configuration."
        )

    splitter = TimeSeriesSplit(
        n_splits=n_splits,
        test_size=test_size,
    )

    fold_results = []

    pooled = {
        "Persistence baseline": {"actual": [], "predicted": []},
        "Linear Regression": {"actual": [], "predicted": []},
        "Random Forest (tuned return)": {
            "actual": [],
            "predicted": [],
        },
    }

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X),
        start=1,
    ):
        X_train, X_test = X[train_idx], X[test_idx]

        y_train, y_test = y[train_idx], y[test_idx]

        close_train = current_close[train_idx]
        close_test = current_close[test_idx]

        print(f"\nWalk-forward fold {fold}/{n_splits}")
        print(f"Training observations: {len(train_idx)}")
        print(f"Testing observations: {len(test_idx)}")
        print(
            f"Test period: {dates[test_idx[0]]} "
            f"to {dates[test_idx[-1]]}"
        )

        # 1. Persistence baseline
        persistence_pred = close_test

        # 2. Linear Regression
        linear = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])

        linear.fit(X_train, y_train)
        linear_pred = linear.predict(X_test)

        # 3. Random Forest predicting percentage returns
        train_returns = (
            (y_train - close_train) / close_train
        )

        return_model = RandomForestRegressor(
            **return_params,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        return_model.fit(X_train, train_returns)

        predicted_returns = return_model.predict(X_test)

        return_pred = close_test * (1 + predicted_returns)

        fold_predictions = {
            "Persistence baseline": persistence_pred,
            "Linear Regression": linear_pred,
            "Random Forest (tuned return)": return_pred,
        }

        for model_name, prediction in fold_predictions.items():

            metrics = calculate_metrics(y_test, prediction)

            fold_results.append({
                "Fold": fold,
                "Model": model_name,
                "Train_Rows": len(train_idx),
                "Test_Rows": len(test_idx),
                "Test_Start": str(dates[test_idx[0]]),
                "Test_End": str(dates[test_idx[-1]]),
                **metrics,
            })

            pooled[model_name]["actual"].extend(y_test.tolist())
            pooled[model_name]["predicted"].extend(
                prediction.tolist()
            )

    # Save individual fold results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fold_df = pl.DataFrame(fold_results)

    fold_df.write_csv(
        output_path / "walk_forward_fold_results.csv"
    )

    # Calculate overall metrics using all walk-forward predictions
    summary_rows = []

    for model_name, values in pooled.items():

        metrics = calculate_metrics(
            values["actual"],
            values["predicted"],
        )

        summary_rows.append({
            "Model": model_name,
            "Folds": n_splits,
            "Total_Test_Observations": len(values["actual"]),
            **metrics,
        })

    summary_df = pl.DataFrame(summary_rows).sort("MAE")

    summary_df.write_csv(
        output_path / "walk_forward_summary.csv"
    )

    print("\n========== WALK-FORWARD FOLD RESULTS ==========")
    print(fold_df)

    print("\n========== WALK-FORWARD OVERALL RESULTS ==========")
    print(summary_df)

    return fold_df, summary_df

def train(
    df: pl.DataFrame,
    output_dir: str = "outputs",
    model_dir: str = "models",
):
    """Train, evaluate, and save AAPL forecasting experiments."""

    featured = make_features(df)

    if featured.height < 150:
        raise ValueError(
            "Not enough feature-complete observations for a reliable split."
        )

    # ---------------------------------------------------------
    # 1. Chronological train/test split
    # ---------------------------------------------------------

    split_at = int(featured.height * 0.80)

    train_df = featured[:split_at]
    test_df = featured[split_at:]

    X_train = train_df.select(FEATURE_COLUMNS).to_numpy()
    y_train = train_df[TARGET_COLUMN].to_numpy()

    X_test = test_df.select(FEATURE_COLUMNS).to_numpy()
    y_test = test_df[TARGET_COLUMN].to_numpy()

    current_test_close = test_df["Close"].to_numpy()
    target_dates = test_df["Target_Date"].to_list()

    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError("Non-finite values found in model features.")

    if not np.isfinite(y_train).all() or not np.isfinite(y_test).all():
        raise ValueError("Non-finite values found in target values.")

    # ---------------------------------------------------------
    # 2. Output directories
    # ---------------------------------------------------------

    output_path = Path(output_dir)
    figures_path = output_path / "figures"
    model_path = Path(model_dir)

    figures_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 3. Dataset and target diagnostics
    # ---------------------------------------------------------

    print("\n========== DATASET DIAGNOSTICS ==========")

    print(f"Total feature-complete observations: {featured.height}")
    print(f"Training observations: {len(X_train)}")
    print(f"Testing observations: {len(X_test)}")

    print(
        "\nTraining period:",
        train_df["Date"].min(),
        "to",
        train_df["Date"].max(),
    )

    print(
        "Testing period:",
        test_df["Date"].min(),
        "to",
        test_df["Date"].max(),
    )

    print("\nTraining target statistics:")
    print(train_df[TARGET_COLUMN].describe())

    print("\nTesting target statistics:")
    print(test_df[TARGET_COLUMN].describe())

    train_min = float(np.min(y_train))
    train_max = float(np.max(y_train))

    test_min = float(np.min(y_test))
    test_max = float(np.max(y_test))

    print("\nTarget range comparison:")
    print(f"Training: {train_min:.2f} to {train_max:.2f}")
    print(f"Testing:  {test_min:.2f} to {test_max:.2f}")

    print(
        "Test observations outside training target range:",
        int(np.sum((y_test < train_min) | (y_test > train_max))),
        "/",
        len(y_test),
    )

    diagnostic_rows = [
        {
            "Dataset": "Training",
            "Rows": len(y_train),
            "Target_Min": train_min,
            "Target_Max": train_max,
            "Target_Mean": float(np.mean(y_train)),
            "Target_Std": float(np.std(y_train)),
            "Start_Date": str(train_df["Date"].min()),
            "End_Date": str(train_df["Date"].max()),
        },
        {
            "Dataset": "Testing",
            "Rows": len(y_test),
            "Target_Min": test_min,
            "Target_Max": test_max,
            "Target_Mean": float(np.mean(y_test)),
            "Target_Std": float(np.std(y_test)),
            "Start_Date": str(test_df["Date"].min()),
            "End_Date": str(test_df["Date"].max()),
        },
    ]

    pl.DataFrame(diagnostic_rows).write_csv(
        output_path / "target_diagnostics.csv"
    )

    # ---------------------------------------------------------
    # 4. Persistence baseline
    # ---------------------------------------------------------

    results = []
    predictions = {}

    persistence_pred = current_test_close

    results.append({
        "Experiment": "Persistence baseline",
        **calculate_metrics(y_test, persistence_pred),
    })

    predictions["Persistence baseline"] = persistence_pred

    # ---------------------------------------------------------
    # 5. Linear Regression
    # ---------------------------------------------------------

    linear = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])

    linear.fit(X_train, y_train)

    linear_pred = linear.predict(X_test)

    results.append({
        "Experiment": "Linear Regression",
        **calculate_metrics(y_test, linear_pred),
    })

    predictions["Linear Regression"] = linear_pred

    # ---------------------------------------------------------
    # 6. Default Random Forest: absolute closing price
    # ---------------------------------------------------------

    rf_default = RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    rf_default.fit(X_train, y_train)

    rf_default_pred = rf_default.predict(X_test)

    results.append({
        "Experiment": "Random Forest (default)",
        **calculate_metrics(y_test, rf_default_pred),
    })

    predictions["Random Forest (default)"] = rf_default_pred

    # ---------------------------------------------------------
    # 7. Tuned Random Forest: absolute closing price
    # ---------------------------------------------------------

    print("\n========== TUNING PRICE MODEL ==========")

    price_search = tune_random_forest(X_train, y_train)

    price_model = price_search.best_estimator_
    price_pred = price_model.predict(X_test)

    results.append({
        "Experiment": "Random Forest (tuned price)",
        **calculate_metrics(y_test, price_pred),
    })

    predictions["Random Forest (tuned price)"] = price_pred

    # ---------------------------------------------------------
    # 8. Return-based Random Forest
    # ---------------------------------------------------------

    print("\n========== TRAINING RETURN MODEL ==========")

    # Predict the percentage change from today's Close
    # to the next trading session's Close.
    y_train_return = (
        (y_train - train_df["Close"].to_numpy())
        / train_df["Close"].to_numpy()
    )

    y_test_return = (
        (y_test - current_test_close)
        / current_test_close
    )

    return_search = tune_random_forest(
        X_train,
        y_train_return,
    )

    return_model = return_search.best_estimator_

    predicted_return = return_model.predict(X_test)

    # Convert predicted returns back into closing prices.
    return_pred = current_test_close * (1 + predicted_return)

    results.append({
        "Experiment": "Random Forest (tuned return)",
        **calculate_metrics(y_test, return_pred),
    })

    predictions["Random Forest (tuned return)"] = return_pred

    # Walk-forward evaluation using the return model's
    # hyperparameters selected on the initial training period.
    walk_forward_evaluation(
    featured=featured,
    return_params=return_search.best_params_,
    output_dir=output_path,
    n_splits=5,
    test_size=250,
    )

    predictions["Random Forest (tuned return)"] = return_pred

    # ---------------------------------------------------------
    # 9. Save experiment results
    # ---------------------------------------------------------

    results_df = pl.DataFrame(results).sort("MAE")

    results_df.write_csv(
        output_path / "experiment_results.csv"
    )

    print("\n========== MODEL COMPARISON ==========")
    print(results_df)

    # ---------------------------------------------------------
    # 10. Save predictions for further analysis
    # ---------------------------------------------------------

    prediction_data = {
        "Target_Date": target_dates,
        "Actual_Close": y_test,
        "Persistence_Prediction": persistence_pred,
        "Linear_Regression_Prediction": linear_pred,
        "Random_Forest_Default_Prediction": rf_default_pred,
        "Random_Forest_Tuned_Price_Prediction": price_pred,
        "Random_Forest_Tuned_Return_Prediction": return_pred,
    }

    pl.DataFrame(prediction_data).write_csv(
        output_path / "test_predictions.csv"
    )

    # ---------------------------------------------------------
    # 11. Actual versus predicted plots
    # ---------------------------------------------------------

    save_prediction_plot(
        target_dates,
        y_test,
        price_pred,
        "AAPL: tuned Random Forest — absolute closing price",
        figures_path / "actual_vs_predicted_price.png",
    )

    save_prediction_plot(
        target_dates,
        y_test,
        return_pred,
        "AAPL: tuned Random Forest — return-based prediction",
        figures_path / "actual_vs_predicted_return.png",
    )

    # ---------------------------------------------------------
    # 12. Feature importance
    # ---------------------------------------------------------

    importance = pl.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": price_model.feature_importances_,
    }).sort("Importance", descending=True)

    importance.write_csv(
        output_path / "feature_importance.csv"
    )

    plt.figure(figsize=(9, 6))

    plt.barh(
        importance["Feature"].to_list()[::-1],
        importance["Importance"].to_list()[::-1],
    )

    plt.title("Tuned Random Forest feature importance")
    plt.xlabel("Importance")
    plt.tight_layout()

    plt.savefig(
        figures_path / "feature_importance.png",
        dpi=160,
    )

    plt.close()

    # ---------------------------------------------------------
    # 13. Save both trained models
    # ---------------------------------------------------------

    joblib.dump(
        price_model,
        model_path / "aapl_close_model.joblib",
    )

    joblib.dump(
        return_model,
        model_path / "aapl_return_model.joblib",
    )

    # ---------------------------------------------------------
    # 14. Save model metadata
    # ---------------------------------------------------------

    metadata = {
        "ticker": "AAPL",
        "forecast_horizon": "next available trading observation",
        "feature_columns": FEATURE_COLUMNS,
        "price_model": {
            "filename": "aapl_close_model.joblib",
            "target": TARGET_COLUMN,
            "best_params": price_search.best_params_,
        },
        "return_model": {
            "filename": "aapl_return_model.joblib",
            "target": "next_session_percentage_return",
            "formula": "(next_close - current_close) / current_close",
            "reconstruction": "current_close * (1 + predicted_return)",
            "best_params": return_search.best_params_,
        },
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_start_target_date": str(test_df["Target_Date"].min()),
        "test_end_target_date": str(test_df["Target_Date"].max()),
    }

    (model_path / "aapl_model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # 15. Final summary
    # ---------------------------------------------------------

    print("\n========== MODEL ARTIFACTS ==========")
    print(f"Results: {output_path / 'experiment_results.csv'}")
    print(f"Predictions: {output_path / 'test_predictions.csv'}")
    print(f"Diagnostics: {output_path / 'target_diagnostics.csv'}")
    print(f"Models: {model_path}")

    print("\nBest price-model parameters:")
    print(price_search.best_params_)

    print("\nBest return-model parameters:")
    print(return_search.best_params_)

    print("\nPrice-model feature importance:")
    print(importance)

    return results_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        help="Optional local CSV or Parquet path.",
    )

    args = parser.parse_args()

    raw = load_source(args.source)
    aapl = validate_and_clean(raw, "AAPL")

    train(aapl)


if __name__ == "__main__":
    main()