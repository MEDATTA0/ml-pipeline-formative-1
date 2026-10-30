# World Stock Prices: Time-Series Pipeline

Daily prices for 62 companies (2000 to 2025) from
[no-ry/world-stock-prices-daily-updating](https://huggingface.co/datasets/no-ry/world-stock-prices-daily-updating).
We store them in SQLite and MongoDB, serve them through an API, and use an LSTM
model to forecast the next trading day's close and how far the price will swing.

## Setup

You need [uv](https://docs.astral.sh/uv/), Docker and [Git LFS](https://git-lfs.com/).
Run every command from the repository root.

```bash
git lfs pull
uv sync
curl -L -o World-Stock-Prices-Dataset.csv https://huggingface.co/datasets/no-ry/world-stock-prices-daily-updating/resolve/main/World-Stock-Prices-Dataset.csv
```

## Task 1: Analysis and model

Open the notebooks in order, `01_data_cleaning` to `05_final_model`. Notebook 05
saves the final model to `models/lstm_two_head.pt`.
`task1_stock_analysis.ipynb` has the questions on Apple.

```bash
uv run jupyter lab notebooks
```

## Task 2: Databases

```bash
docker compose up -d
uv run python -m formative_1.load_to_db
```

The SQL schema is `src/formative_1/tables.sql`. MongoDB has one collection with one
document per ticker per day. Set `MONGO_URL` if MongoDB is not on
`mongodb://localhost:27017`.

![ERD of the SQLite database](docs/erd.png)

## Task 3: API

```bash
uv run fastapi dev src/formative_1/api.py
```

Docs are at <http://127.0.0.1:8000/docs>.

| Method | Path | What it does |
| --- | --- | --- |
| GET | `/latest` | Newest row of every ticker |
| GET | `/?start=YYYY-MM-DD&end=YYYY-MM-DD` | Rows in a date range |
| POST | `/` | Add a row |
| PUT | `/{ticker}/{day}` | Replace a row |
| DELETE | `/{ticker}/{day}` | Delete a row |

Every endpoint works on both databases, and changes are saved in both.

## Task 4: Forecast

With the API running:

```bash
uv run python -m formative_1.forecast
```

```
Fetched 7094 sqlite records up to 2025-07-03.

========== AAPL NEXT-SESSION FORECAST ==========
Based on session:       2025-07-03
Close on that session:  213.55
Predicted change:       +0.01%
Predicted next close:   213.57
Predicted High/Low gap: 1.79%
```

Options: `--ticker` (default `AAPL`), `--source sqlite|mongodb`, `--api-url`,
`--lookback-days`, `--model-file`.
