import sqlite3
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Query
from pydantic import BaseModel, model_validator
from pymongo import MongoClient
from pymongo.collection import Collection

from formative_1.config import mongo_collection_name, mongo_db_name, mongo_url

app = FastAPI()
db_path = Path("stocks.db")


def get_mongo_collection() -> Iterator[Collection[Any]]:
    mongo_client = MongoClient(mongo_url)
    try:
        yield mongo_client[mongo_db_name][mongo_collection_name]
    finally:
        mongo_client.close()


def get_db() -> Iterator[sqlite3.Connection]:
    db_con = sqlite3.connect(db_path)
    db_con.row_factory = sqlite3.Row
    try:
        yield db_con
    finally:
        db_con.close()


SqliteClient = Annotated[sqlite3.Connection, Depends(get_db)]
MongoCollection = Annotated[Collection[Any], Depends(get_mongo_collection)]

# Same columns as the SQLite queries, without Mongo's _id
mongo_projection = {
    "_id": 0,
    "ticker": 1,
    "date": 1,
    "open": 1,
    "high": 1,
    "low": 1,
    "close": 1,
    "volume": 1,
}


@app.get("/latest")
def get_latest(db_con: SqliteClient, collection: MongoCollection):
    rows = db_con.execute(
        """
        SELECT t."name" AS ticker, s."date", s."open", s."high", s."low",
               s."close", s."volume"
        FROM "stocks" s
        JOIN "tickers" t ON t."id" = s."ticker_id"
        WHERE s."date" = (
            SELECT MAX("date") FROM "stocks" WHERE "ticker_id" = s."ticker_id"
        )
        ORDER BY t."name"
        """
    ).fetchall()

    mongo_rows = collection.aggregate(
        [
            {"$sort": {"ticker": 1, "date": -1}},
            {"$group": {"_id": "$ticker", "doc": {"$first": "$$ROOT"}}},
            {"$replaceRoot": {"newRoot": "$doc"}},
            {"$project": mongo_projection},
            {"$sort": {"ticker": 1}},
        ]
    )

    return {"sqlite": [dict(row) for row in rows], "mongodb": list(mongo_rows)}


class FilterParams(BaseModel):
    start: date
    end: date

    @model_validator(mode="after")
    def check_order(self):
        if self.start > self.end:
            raise ValueError("start must be on or before end")
        return self


@app.get("/")
def get_by_date_range(
    params: Annotated[FilterParams, Query()],
    db_con: SqliteClient,
    collection: MongoCollection,
):
    start, end = params.start.isoformat(), params.end.isoformat()

    rows = db_con.execute(
        """
        SELECT t."name" AS ticker, s."date", s."open", s."high", s."low",
               s."close", s."volume"
        FROM "stocks" s
        JOIN "tickers" t ON t."id" = s."ticker_id"
        WHERE s."date" BETWEEN ? AND ?
        ORDER BY s."date"
        """,
        (start, end),
    ).fetchall()

    mongo_rows = collection.find(
        {"date": {"$gte": start, "$lte": end}}, mongo_projection
    ).sort("date", 1)

    return {"sqlite": [dict(row) for row in rows], "mongodb": list(mongo_rows)}
