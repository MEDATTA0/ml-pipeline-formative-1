CREATE TABLE "countries" (
    "id" INTEGER PRIMARY KEY,
    "name" TEXT NOT NULL UNIQUE
);

CREATE TABLE "industries" (
    "id" INTEGER PRIMARY KEY,
    -- "name" TEXT NOT NULL UNIQUE,
    "tag" TEXT NOT NULL UNIQUE
);

CREATE TABLE "companies" (
    "id" INTEGER PRIMARY KEY,
    "name" TEXT NOT NULL,
    "country_id" INTEGER NOT NULL,
    "industry_id" INTEGER NOT NULL,
    FOREIGN KEY ("country_id") REFERENCES "countries"("id"),
    FOREIGN KEY ("industry_id") REFERENCES "industries"("id")
);


CREATE TABLE "tickers" (
    "id" INTEGER PRIMARY KEY,
    "name" TEXT NOT NULL UNIQUE,
    "company_id" INTEGER NOT NULL,
    FOREIGN KEY ("company_id") REFERENCES "companies"("id")
);

CREATE TABLE "stocks" (
    "id" INTEGER PRIMARY KEY,
    "ticker_id" INTEGER NOT NULL,
    "date" DATE NOT NULL,
    "open" REAL NOT NULL,
    "high" REAL NOT NULL,
    "low" REAL NOT NULL,
    "close" REAL NOT NULL,
    "volume" INTEGER NOT NULL,
    "dividends" REAL NOT NULL,
    "stock_split" REAL NOT NULL,
    "capital_gains" REAL NOT NULL,
    FOREIGN KEY ("ticker_id") REFERENCES "tickers"("id")
);

CREATE UNIQUE INDEX "stocks_ticker_date" ON "stocks" ("ticker_id", "date");
