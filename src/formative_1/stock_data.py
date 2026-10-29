"""Cleaning, features and windowing for the next-day return and volatility models.

The notebooks and the prediction script share these functions, so training and
inference apply exactly the same preprocessing.
"""

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

WINDOW = 30  # trading days of history the model reads
FEATURES = ["log_ret", "vol_dev", "hl_range", "mkt_ret"]
VAL_START = date(2019, 1, 1)
TEST_START = date(2022, 1, 1)

VOLUME_DAYS = 60  # rolling window for a ticker's usual volume
VOLUME_MIN_DAYS = 30  # days of volume history needed before `vol_dev` exists
SPLIT_RATIO = 5  # a one-day price drop beyond this factor is a stock split

# The CSV lives at the repository root, where load_to_db.py also reads it.
DATA_PATH = Path(__file__).resolve().parents[2] / "World-Stock-Prices-Dataset.csv"


def read_raw(path: Path = DATA_PATH) -> pl.DataFrame:
    """Read the CSV and add `day`, the calendar date of each row."""
    return pl.read_csv(path).with_columns(
        day=pl.col("Date").str.slice(0, 10).str.to_date()
    )


def dedupe(raw: pl.DataFrame) -> pl.DataFrame:
    """Keep one row per (Date, Ticker); the daily-updating source sometimes records a day twice."""
    return raw.unique(subset=["Date", "Ticker"], keep="last", maintain_order=True)


def drop_untraded(data: pl.DataFrame) -> pl.DataFrame:
    """Drop zero-volume rows (stale carried-forward quotes) and sort by ticker and day."""
    return data.filter(pl.col("Volume") > 0).sort("Ticker", "day")


def find_seams(traded: pl.DataFrame) -> pl.DataFrame:
    """Days where a ticker's price drops by a whole-number factor above SPLIT_RATIO in one day.

    These are forward stock splits whose earlier history is on a different basis.
    """
    return (
        traded.with_columns(
            ratio=pl.col("Close").shift(1).over("Ticker") / pl.col("Close")
        )
        .filter(pl.col("ratio") > SPLIT_RATIO)
        .select("Ticker", seam_day="day", factor=pl.col("ratio").round())
    )


def adjust_splits(traded: pl.DataFrame, seams: pl.DataFrame) -> pl.DataFrame:
    """Put the history before each seam on the post-split basis."""
    adjust = (
        pl.when(pl.col("day") < pl.col("seam_day"))
        .then(pl.col("factor"))
        .otherwise(1.0)
    ).fill_null(1.0)
    return (
        traded.join(seams, on="Ticker", how="left")
        .with_columns(adjust=adjust)
        .with_columns(
            pl.col("Open", "High", "Low", "Close") / pl.col("adjust"),
            pl.col("Volume") * pl.col("adjust"),
        )
        .drop("seam_day", "factor", "adjust")
    )


def clean(raw: pl.DataFrame) -> pl.DataFrame:
    """Full cleaning pipeline: dedupe, drop untraded days, fix split seams."""
    traded = drop_untraded(dedupe(raw))
    return adjust_splits(traded, find_seams(traded))


def add_features(data: pl.DataFrame) -> pl.DataFrame:
    """Add the model inputs and two targets, both for the next trading day:
    `target` (log return, used to forecast the close price) and `target_vol`
    (log(High/Low), the intraday range, used as the volatility measure).

    `mkt_ret` is the equal-weighted mean log return of every ticker on that day.
    """
    log_volume = pl.col("Volume").log()
    usual_log_volume = log_volume.rolling_mean(
        VOLUME_DAYS, min_samples=VOLUME_MIN_DAYS
    ).over("Ticker")
    data = data.with_columns(
        log_ret=pl.col("Close").log().diff().over("Ticker"),
        vol_dev=log_volume - usual_log_volume,
        hl_range=(pl.col("High") / pl.col("Low")).log(),
    )
    market = data.group_by("day").agg(mkt_ret=pl.col("log_ret").mean())
    return (
        data.join(market, on="day")
        .sort("Ticker", "day")
        .with_columns(
            target=pl.col("log_ret").shift(-1).over("Ticker"),
            target_vol=pl.col("hl_range").shift(-1).over("Ticker"),
        )
    )


def _ticker_frames(data: pl.DataFrame) -> Iterator[tuple[str, pl.DataFrame]]:
    """Each ticker's rows with complete features, skipping tickers shorter than a window."""
    for (ticker,), frame in data.partition_by("Ticker", as_dict=True).items():
        frame = frame.drop_nulls(FEATURES)  # drops the warm-up rows of each ticker
        if frame.height >= WINDOW:
            yield ticker, frame


def make_windows(data: pl.DataFrame, extras: tuple[str, ...] = ()):
    """Return (X, y, target_day): one window of the last WINDOW days per sample.

    `y` is the next-day log return. Each column named in `extras` (for example "Ticker",
    "Close" or "target_vol") is returned as one more array, taken at the window's last row.

    `target_day` is the day being predicted (the day after the window's last row),
    so splitting on it never lets a training target fall inside the test period.
    """
    xs, ys, days = [], [], []
    extra_cols = {column: [] for column in extras}
    for _, frame in _ticker_frames(data):
        values = frame.select(FEATURES).to_numpy().astype(np.float32)
        windows = sliding_window_view(values, WINDOW, axis=0).transpose(0, 2, 1)

        # Window i ends at row i + WINDOW - 1, and every other value is read at that row.
        at_window_end = slice(WINDOW - 1, None)
        target = frame["target"].to_numpy()[at_window_end]
        has_target = ~np.isnan(target)  # the last row of a ticker has no next day

        xs.append(windows[has_target])
        ys.append(target[has_target].astype(np.float32))
        days.append(frame["day"].shift(-1).to_numpy()[at_window_end][has_target])
        for column in extras:
            extra_cols[column].append(
                frame[column].to_numpy()[at_window_end][has_target]
            )

    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(days),
        *(np.concatenate(extra_cols[column]) for column in extras),
    )


def split_masks(day):
    """Boolean masks that cut the samples by the date of their target."""
    train = day < np.datetime64(VAL_START)
    val = (day >= np.datetime64(VAL_START)) & (day < np.datetime64(TEST_START))
    test = day >= np.datetime64(TEST_START)
    return train, val, test


def window_summary(X):
    """Flatten windows into a compact table for non-sequence models.

    Returns (table, names): the last 5 days of every feature, the mean and standard
    deviation of `log_ret` over the last 5/10/30 days, and the mean `hl_range` over 5/30.
    """
    columns, names = [], []

    for lag in range(5):  # lag 0 is the most recent day
        for k, feature in enumerate(FEATURES):
            columns.append(X[:, -1 - lag, k])
            names.append(f"{feature}_lag{lag}")

    log_ret = X[:, :, FEATURES.index("log_ret")]
    for days in (5, 10, 30):
        columns += [log_ret[:, -days:].mean(axis=1), log_ret[:, -days:].std(axis=1)]
        names += [f"ret_mean_{days}", f"ret_std_{days}"]

    hl_range = X[:, :, FEATURES.index("hl_range")]
    for days in (5, 30):
        columns.append(hl_range[:, -days:].mean(axis=1))
        names.append(f"hl_mean_{days}")

    return np.column_stack(columns).astype(np.float32), names


def latest_windows(data: pl.DataFrame):
    """The most recent window of every ticker, for forecasting the next trading day.

    `data` comes from `add_features`. Returns (X, tickers, last_day, last_close), one entry per
    ticker with at least WINDOW usable rows. Unlike `make_windows`, the final row is kept even
    though its target is not known yet.
    """
    windows, tickers, last_days, last_closes = [], [], [], []
    for ticker, frame in _ticker_frames(data):
        last = frame.tail(WINDOW)
        windows.append(last.select(FEATURES).to_numpy().astype(np.float32))
        tickers.append(ticker)
        last_days.append(last["day"][-1])
        last_closes.append(last["Close"][-1])
    return (
        np.stack(windows),
        np.array(tickers),
        np.array(last_days),
        np.array(last_closes),
    )
