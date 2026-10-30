import os
import sqlite3
from pathlib import Path

import polars as pl
from datasets import load_dataset
from pymongo import ASCENDING, MongoClient

from formative_1.config import mongo_collection_name, mongo_db_name, mongo_url

df_filename = "World-Stock-Prices-Dataset.csv"
db_path = "stocks.db"
schema_path = Path(__file__).with_name("tables.sql")


def download_dataset() -> None:
    if not os.path.exists(df_filename):
        load_dataset("no-ry/world-stock-prices-daily-updating")
        print(f"Dataset downloaded to {os.path.curdir + '/' + df_filename}")
    else:
        print("Dataset already exists")


def read_dataset() -> pl.DataFrame:
    df = pl.read_csv(df_filename)
    print(df.shape)
    print(df.columns)

    return df.with_columns(
        pl.col("Date").str.slice(0, 10),
        # Replace null with 0.0. In our case, emptyness means 0.0
        pl.col("Capital Gains").fill_null(0.0),
    ).unique(subset=["Date", "Ticker"], keep="last", maintain_order=True)


def build_tables(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    # One row per entity, keyed by a 1-based surrogate id
    countries = df.select(name="Country").unique().sort("name").with_row_index("id", 1)
    industries = (
        df.select(tag="Industry_Tag").unique().sort("tag").with_row_index("id", 1)
    )
    companies = (
        df.select("Brand_Name", "Country", "Industry_Tag")
        .unique()
        .sort("Brand_Name")
        .with_row_index("id", 1)
        .join(
            countries.rename({"id": "country_id"}), left_on="Country", right_on="name"
        )
        .join(
            industries.rename({"id": "industry_id"}),
            left_on="Industry_Tag",
            right_on="tag",
        )
        .select(
            "id", name="Brand_Name", country_id="country_id", industry_id="industry_id"
        )
    )
    tickers = (
        df.select("Ticker", "Brand_Name")
        .unique()
        .sort("Ticker")
        .with_row_index("id", 1)
        .join(companies.select(company_id="id", Brand_Name="name"), on="Brand_Name")
        .select("id", name="Ticker", company_id="company_id")
    )
    stocks = df.join(tickers.select(ticker_id="id", Ticker="name"), on="Ticker").select(
        "ticker_id",
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        pl.col("Volume").cast(pl.Int64),
        "Dividends",
        "Stock Splits",
        "Capital Gains",
    )

    return {
        "countries": countries,
        "industries": industries,
        "companies": companies,
        "tickers": tickers,
        "stocks": stocks,
    }


def create_database() -> sqlite3.Connection:
    if os.path.exists(db_path):
        os.remove(db_path)

    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(schema_path.read_text())
    return con


def insert(
    con: sqlite3.Connection, table: str, columns: list[str], frame: pl.DataFrame
) -> None:
    placeholders = ", ".join("?" * len(columns))
    cols = ", ".join(f'"{c}"' for c in columns)
    con.executemany(
        f'INSERT INTO "{table}" ({cols}) VALUES ({placeholders})',
        frame.iter_rows(),
    )
    print(f"{table}: {frame.height} rows")


def load_tables(con: sqlite3.Connection, tables: dict[str, pl.DataFrame]) -> None:
    with con:
        insert(con, "countries", ["id", "name"], tables["countries"])
        insert(con, "industries", ["id", "tag"], tables["industries"])
        insert(
            con,
            "companies",
            ["id", "name", "country_id", "industry_id"],
            tables["companies"],
        )
        insert(con, "tickers", ["id", "name", "company_id"], tables["tickers"])
        insert(
            con,
            "stocks",
            [
                "ticker_id",
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "dividends",
                "stock_split",
                "capital_gains",
            ],
            tables["stocks"],
        )


def load_mongo(df: pl.DataFrame) -> None:
    # Denormalised: one document per stock row, so no joins are needed at query time
    docs = df.select(
        ticker="Ticker",
        company="Brand_Name",
        country="Country",
        industry="Industry_Tag",
        date="Date",
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume=pl.col("Volume").cast(pl.Int64),
        dividends="Dividends",
        stock_split="Stock Splits",
        capital_gains="Capital Gains",
    )

    client = MongoClient(mongo_url)
    try:
        collection = client[mongo_db_name][mongo_collection_name]
        collection.drop()
        for batch in docs.iter_slices(50_000):
            collection.insert_many(batch.to_dicts())
        collection.create_index([("date", ASCENDING)])
        collection.create_index([("ticker", ASCENDING), ("date", ASCENDING)])
        print(f"mongodb {mongo_collection_name}: {docs.height} documents")
    finally:
        client.close()


def main() -> None:
    download_dataset()
    df = read_dataset()
    tables = build_tables(df)

    con = create_database()
    try:
        load_tables(con, tables)
    finally:
        con.close()
    print(f"Loaded into {db_path}")

    load_mongo(df)


if __name__ == "__main__":
    main()
