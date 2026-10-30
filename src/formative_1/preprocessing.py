"""Load and validate the stock-price data used by Task 1."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets

DATASET_ID = "no-ry/world-stock-prices-daily-updating"

REQUIRED_COLUMNS = [
    "Date", "Ticker", "Open", "High", "Low", "Close", "Volume",
    "Dividends", "Stock Splits",
]

NUMERIC_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "Dividends", "Stock Splits",
]


def _dataset_to_polars(dataset_id: str = DATASET_ID) -> pl.DataFrame:
    """Load a Hugging Face dataset without assuming a particular split name."""
    loaded = load_dataset(dataset_id)

    if isinstance(loaded, DatasetDict):
        splits = list(loaded.values())

        if not splits:
            raise ValueError(
                f"No splits found in Hugging Face dataset {dataset_id!r}"
            )

        dataset = (
            splits[0]
            if len(splits) == 1
            else concatenate_datasets(splits)
        )

    elif isinstance(loaded, Dataset):
        dataset = loaded

    else:
        raise TypeError(f"Unexpected dataset object: {type(loaded)!r}")

    return dataset.to_polars()


def load_source(source: str | None = None) -> pl.DataFrame:
    """Read CSV/Parquet locally, or load the configured Hugging Face dataset."""
    if source:
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(path)

        suffix = path.suffix.lower()

        if suffix == ".csv":
            return pl.read_csv(
                path,
                try_parse_dates=True,
                infer_schema_length=10000,
            )

        if suffix in {".parquet", ".pq"}:
            return pl.read_parquet(path)

        raise ValueError("Source must be a .csv or .parquet file.")

    return _dataset_to_polars()


def _reconcile_duplicate_dates(df: pl.DataFrame) -> pl.DataFrame:
    """Consolidate duplicate daily observations using a documented rule.

    Closing prices must agree across duplicate records. Other numeric
    market fields are reconciled using their median. Descriptive metadata
    retains the first available value.
    """
    duplicate_dates = (
        df.group_by("Date")
        .len()
        .filter(pl.col("len") > 1)
        .sort("Date")
    )

    if duplicate_dates.is_empty():
        return df

    print(
        f"Found {duplicate_dates.height} dates with duplicate observations."
    )

    # Verify that duplicate records do not disagree on the target.
    conflicting_closes = (
        df.group_by("Date")
        .agg(pl.col("Close").n_unique().alias("unique_close"))
        .filter(pl.col("unique_close") > 1)
    )

    if not conflicting_closes.is_empty():
        examples = conflicting_closes.head(10).to_dicts()

        raise ValueError(
            "Duplicate dates contain conflicting Close prices. "
            "Cannot safely consolidate the forecasting target. "
            f"Example dates: {examples}"
        )

    numeric_columns = [
        column for column in NUMERIC_COLUMNS
        if column in df.columns
    ]

    metadata_columns = [
        column for column in df.columns
        if column not in numeric_columns and column != "Date"
    ]

    # Reconcile numeric measurements and preserve descriptive metadata.
    df = (
        df.group_by("Date")
        .agg(
            [
                pl.col(column).median().alias(column)
                for column in numeric_columns
            ]
            + [
                pl.col(column).first().alias(column)
                for column in metadata_columns
            ]
        )
        .sort("Date")
    )

    print(
        f"Consolidated duplicate dates. "
        f"Remaining observations: {df.height:,}"
    )

    return df


def validate_and_clean(
    raw: pl.DataFrame,
    ticker: str = "AAPL",
) -> pl.DataFrame:
    """Normalize types, remove exact duplicates, and clean one ticker."""
    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in raw.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}"
        )

    df = raw.with_columns(
        pl.col("Date")
        .cast(pl.String)
        .str.slice(0, 10)
        .str.to_date(strict=False),

        pl.col("Ticker")
        .cast(pl.String)
        .str.to_uppercase()
        .str.strip_chars(),

        *[
            pl.col(column).cast(pl.Float64, strict=False)
            for column in NUMERIC_COLUMNS
        ],
    ).filter(
        pl.col("Ticker") == ticker.upper()
    )

    if df.is_empty():
        raise ValueError(
            f"No rows found for ticker {ticker!r}."
        )

    # Remove fully identical records first.
    df = df.unique(maintain_order=True).sort("Date")

    # Exclude records without a valid date or prediction target.
    df = df.filter(
        pl.col("Date").is_not_null()
        & pl.col("Close").is_not_null()
    )

    if df.is_empty():
        raise ValueError(
            f"No usable observations found for ticker {ticker!r}."
        )

    # Consolidate repeated dates after checking target consistency.
    df = _reconcile_duplicate_dates(df)

    if df.height < 100:
        raise ValueError(
            f"Only {df.height} usable {ticker} observations were found."
        )

    # Validate OHLC relationships after reconciliation.
    invalid_ohlc = df.filter(
        (pl.col("High") < pl.col("Low"))
        | (pl.col("High") < pl.col("Open"))
        | (pl.col("High") < pl.col("Close"))
        | (pl.col("Low") > pl.col("Open"))
        | (pl.col("Low") > pl.col("Close"))
    )

    if not invalid_ohlc.is_empty():
        examples = invalid_ohlc.select(
            ["Date", "Open", "High", "Low", "Close"]
        ).head(10).to_dicts()

        raise ValueError(
            "Invalid OHLC relationships found after duplicate "
            f"reconciliation. Examples: {examples}"
        )

    # Guarantee one observation per trading date.
    remaining_duplicates = (
        df.group_by("Date")
        .len()
        .filter(pl.col("len") > 1)
    )

    if not remaining_duplicates.is_empty():
        raise ValueError(
            "Duplicate dates remain after reconciliation."
        )

    return df.sort("Date")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        help="Optional local CSV or Parquet path.",
    )

    parser.add_argument(
        "--ticker",
        default="AAPL",
    )

    parser.add_argument(
        "--output",
        default="data/processed/aapl_clean.parquet",
    )

    args = parser.parse_args()

    raw = load_source(args.source)

    clean = validate_and_clean(raw, args.ticker)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    clean.write_parquet(output)

    print("\nPreprocessing completed successfully.")
    print(f"Ticker: {args.ticker.upper()}")
    print(f"Rows saved: {clean.height:,}")
    print(f"Output: {output}")
    print(
        f"Date range: {clean['Date'].min()} "
        f"to {clean['Date'].max()}"
    )


if __name__ == "__main__":
    main()