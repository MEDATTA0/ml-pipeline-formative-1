"""Task 4: forecast a ticker's next close from records served by the API."""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import polars as pl
import torch

from formative_1 import stock_data as sd
from formative_1.lstm_model import DEVICE, TwoHeadLSTM, predict

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL_FILE = "models/lstm_two_head.pt"

# The model reads 30 sessions, and the volume feature in the first of them
# looks back 60 more, so 180 calendar days (about 120 sessions) covers both.
DEFAULT_LOOKBACK_DAYS = 180

SOURCES = ("sqlite", "mongodb")

# API field names mapped to the dataset column names used in training.
API_TO_DATASET_COLUMNS = {
    "ticker": "Ticker",
    "date": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}


def _get_json(url: str) -> dict[str, Any]:
    """GET a URL and decode its JSON body."""
    try:
        with urlopen(url, timeout=120) as response:
            return json.load(response)
    except URLError as error:
        raise ConnectionError(
            f"Could not reach the API at {url}: {error}"
        ) from error


def fetch_latest_date(api_url: str, ticker: str, source: str) -> date:
    """Return the date of the ticker's most recent record from GET /latest."""
    payload = _get_json(f"{api_url}/latest")

    for record in payload[source]:
        if record["ticker"] == ticker:
            return date.fromisoformat(record["date"])

    raise ValueError(f"The API returned no {source} records for {ticker!r}.")


def fetch_history(
    api_url: str,
    source: str,
    end: date,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> list[dict[str, Any]]:
    """Return every ticker's records from GET /?start=&end=.

    All tickers are kept because one model input, the market return, is the
    average return of every ticker on the same day.
    """
    start = end - timedelta(days=lookback_days)
    query = urlencode({"start": start.isoformat(), "end": end.isoformat()})
    records = _get_json(f"{api_url}/?{query}")[source]

    if not records:
        raise ValueError(
            f"The API returned no {source} records between {start} and {end}."
        )

    return records


def preprocess(records: list[dict[str, Any]], ticker: str):
    """Apply the training cleaning and features; return the ticker's last window."""
    frame = (
        pl.DataFrame(records)
        .rename(API_TO_DATASET_COLUMNS)
        .with_columns(day=pl.col("Date").str.to_date())
    )
    X, tickers, last_days, last_closes = sd.latest_windows(
        sd.add_features(sd.clean(frame))
    )

    if ticker not in tickers:
        raise ValueError(
            f"Not enough history for {ticker!r} to fill a {sd.WINDOW}-session "
            "window. Increase --lookback-days."
        )

    i = int(np.flatnonzero(tickers == ticker)[0])
    return X[i], last_days[i], float(last_closes[i])


def load_model(model_file: Path) -> tuple[dict[str, Any], list[TwoHeadLSTM]]:
    """Load the saved LSTM ensemble and its scaling statistics."""
    if not model_file.exists():
        raise FileNotFoundError(
            f"Trained model file not found: {model_file}. "
            "Run notebooks/05_final_model.ipynb first."
        )

    # The model is stored with Git LFS; without it, the clone has a pointer.
    if model_file.read_bytes()[:40].startswith(b"version https://git-lfs"):
        raise FileNotFoundError(
            f"{model_file} is a Git LFS pointer. Install git-lfs and run "
            "`git lfs pull` to download the model."
        )

    checkpoint = torch.load(model_file, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]
    members = []

    for state in checkpoint["states"]:
        member = TwoHeadLSTM(config["n_features"], hidden=config["hidden"])
        member.load_state_dict(state)
        # predict() sends the inputs to DEVICE, so the weights must be there too.
        member.to(DEVICE)
        members.append(member)

    return checkpoint, members


def forecast_next_close(window, checkpoint, members) -> tuple[float, float]:
    """Return the predicted next-day log return and log(High/Low) range.

    Each member predicts scaled values; the ensemble averages them.
    """
    mean = np.array(checkpoint["mean"], dtype=np.float32)
    std = np.array(checkpoint["std"], dtype=np.float32)
    X = torch.from_numpy(((window - mean) / std)[None])

    scaled = np.mean([predict(member, X)[0] for member in members], axis=0)
    return (
        float(scaled[0] * checkpoint["y_std"]),
        float(scaled[1] * checkpoint["vol_std"]),
    )


def run(
    api_url: str,
    ticker: str,
    source: str,
    lookback_days: int,
    model_file: Path,
) -> dict[str, Any]:
    """Fetch from the API, preprocess, load the model, and forecast."""
    checkpoint, members = load_model(model_file)

    end = fetch_latest_date(api_url, ticker, source)
    history = fetch_history(api_url, source, end, lookback_days)
    print(f"Fetched {len(history)} {source} records up to {end}.")

    window, last_day, close = preprocess(history, ticker)
    log_return, log_range = forecast_next_close(window, checkpoint, members)

    return {
        "ticker": ticker,
        "as_of_date": str(last_day),
        "current_close": close,
        "predicted_change_pct": math.expm1(log_return) * 100,
        "predicted_close": close * math.exp(log_return),
        "predicted_range_pct": math.expm1(log_range) * 100,
    }


def positive_int(value: str) -> int:
    """argparse type for a whole number above zero."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be above zero, got {number}")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forecast the next-session close from API records.",
    )

    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="Base URL of the running FastAPI app.",
    )

    parser.add_argument(
        "--ticker",
        default="AAPL",
        help="Ticker to forecast.",
    )

    parser.add_argument(
        "--source",
        choices=SOURCES,
        default="sqlite",
        help="Which database's records to use from the API response.",
    )

    parser.add_argument(
        "--lookback-days",
        type=positive_int,
        default=DEFAULT_LOOKBACK_DAYS,
        help="Calendar days of history to fetch before the latest record.",
    )

    parser.add_argument(
        "--model-file",
        default=DEFAULT_MODEL_FILE,
        help="Saved LSTM ensemble from notebooks/05_final_model.ipynb.",
    )

    args = parser.parse_args()

    try:
        result = run(
            api_url=args.api_url.rstrip("/"),
            ticker=args.ticker.strip().upper(),
            source=args.source,
            lookback_days=args.lookback_days,
            model_file=Path(args.model_file),
        )
    except (ConnectionError, FileNotFoundError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")

    print(f"\n========== {result['ticker']} NEXT-SESSION FORECAST ==========")
    print(f"Based on session:       {result['as_of_date']}")
    print(f"Close on that session:  {result['current_close']:.2f}")
    print(f"Predicted change:       {result['predicted_change_pct']:+.2f}%")
    print(f"Predicted next close:   {result['predicted_close']:.2f}")
    print(f"Predicted High/Low gap: {result['predicted_range_pct']:.2f}%")


if __name__ == "__main__":
    main()
