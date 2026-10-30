import sqlite3
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Response
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from formative_1.config import mongo_collection_name, mongo_db_name, mongo_url
from formative_1.schemas import FilterParams, Stock, StockValues

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
    rows = db_con.execute("""
        SELECT t."name" AS ticker, s."date", s."open", s."high", s."low",
               s."close", s."volume"
        FROM "stocks" s
        JOIN "tickers" t ON t."id" = s."ticker_id"
        WHERE s."date" = (
            SELECT MAX("date") FROM "stocks" WHERE "ticker_id" = s."ticker_id"
        )
        ORDER BY t."name"
        """).fetchall()

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


def find_ticker(db_con: sqlite3.Connection, ticker: str) -> sqlite3.Row:
    row = db_con.execute(
        """
        SELECT t."id", c."name" AS company, co."name" AS country, i."tag" AS industry
        FROM "tickers" t
        JOIN "companies" c ON c."id" = t."company_id"
        JOIN "countries" co ON co."id" = c."country_id"
        JOIN "industries" i ON i."id" = c."industry_id"
        WHERE t."name" = ?
        """,
        (ticker,),
    ).fetchone()
    if row is None:
        raise HTTPException(404, f"Unknown ticker {ticker}")
    return row


def stock_not_found(ticker: str, day: date) -> HTTPException:
    return HTTPException(404, f"No stock row for {ticker} on {day.isoformat()}")


# The two stores are written together: SQLite commits only after MongoDB accepted
# the write, so a MongoDB failure leaves both unchanged.
mongo_unavailable = HTTPException(503, "MongoDB write failed; nothing was changed")


@app.post("/", status_code=201)
def create_stock(
    stock: Stock, db_con: SqliteClient, collection: MongoCollection
) -> Stock:
    ticker, day = stock.ticker.strip().upper(), stock.date.isoformat()
    info = find_ticker(db_con, ticker)

    in_sqlite = db_con.execute(
        'SELECT 1 FROM "stocks" WHERE "ticker_id" = ? AND "date" = ?', (info["id"], day)
    ).fetchone()
    if in_sqlite or collection.find_one({"ticker": ticker, "date": day}, {"_id": 1}):
        raise HTTPException(409, f"{ticker} already has a row for {day}")

    values = stock.model_dump(exclude={"ticker", "date"})
    try:
        with db_con:
            db_con.execute(
                """
                INSERT INTO "stocks" ("ticker_id", "date", "open", "high", "low", "close",
                                      "volume", "dividends", "stock_split", "capital_gains")
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (info["id"], day, *values.values()),
            )
            collection.insert_one(
                {
                    "ticker": ticker,
                    "company": info["company"],
                    "country": info["country"],
                    "industry": info["industry"],
                    "date": day,
                    **values,
                }
            )
    except sqlite3.IntegrityError:
        raise HTTPException(409, f"{ticker} already has a row for {day}")
    except PyMongoError:
        raise mongo_unavailable
    return stock.model_copy(update={"ticker": ticker})


@app.put("/{ticker}/{day}")
def replace_stock(
    ticker: str,
    day: date,
    values: StockValues,
    db_con: SqliteClient,
    collection: MongoCollection,
) -> Stock:
    ticker, iso = ticker.strip().upper(), day.isoformat()
    info = find_ticker(db_con, ticker)
    fields = values.model_dump()

    try:
        with db_con:
            updated = db_con.execute(
                f"""
                UPDATE "stocks" SET {", ".join(f'"{c}" = ?' for c in fields)}
                WHERE "ticker_id" = ? AND "date" = ?
                """,
                (*fields.values(), info["id"], iso),
            ).rowcount
            if not updated:
                raise stock_not_found(ticker, day)
            # upsert repairs a document that is missing from MongoDB
            collection.update_one(
                {"ticker": ticker, "date": iso},
                {
                    "$set": fields,
                    "$setOnInsert": {
                        "company": info["company"],
                        "country": info["country"],
                        "industry": info["industry"],
                    },
                },
                upsert=True,
            )
    except PyMongoError:
        raise mongo_unavailable
    return Stock(ticker=ticker, date=day, **fields)


@app.delete("/{ticker}/{day}", status_code=204)
def delete_stock(
    ticker: str, day: date, db_con: SqliteClient, collection: MongoCollection
) -> Response:
    ticker, iso = ticker.strip().upper(), day.isoformat()
    info = find_ticker(db_con, ticker)

    try:
        with db_con:
            deleted = db_con.execute(
                'DELETE FROM "stocks" WHERE "ticker_id" = ? AND "date" = ?',
                (info["id"], iso),
            ).rowcount
            deleted_mongo = collection.delete_many(
                {"ticker": ticker, "date": iso}
            ).deleted_count
    except PyMongoError:
        raise mongo_unavailable
    if not deleted and not deleted_mongo:
        raise stock_not_found(ticker, day)
    return Response(status_code=204)
