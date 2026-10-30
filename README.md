# World Stock Prices: Time-Series Pipeline

Daily prices for 62 companies (2000 to 2025) from
[no-ry/world-stock-prices-daily-updating](https://huggingface.co/datasets/no-ry/world-stock-prices-daily-updating),
stored in SQLite and MongoDB, served by an API, and used to forecast the next
trading day.

## Setup

Requires [uv](https://docs.astral.sh/uv/), Docker and
[Git LFS](https://git-lfs.com/). Run every command from the repository root.

```bash
git lfs pull
```

```bash
uv sync
```

```bash
curl -L -o World-Stock-Prices-Dataset.csv https://huggingface.co/datasets/no-ry/world-stock-prices-daily-updating/resolve/main/World-Stock-Prices-Dataset.csv
```

## Task 1: Analysis and model

Open the notebooks in order: `01_data_cleaning` to `05_final_model`.
Notebook `05` trains the final model and saves it to `models/lstm_two_head.pt`
(already committed). `task1_stock_analysis.ipynb` holds the AAPL questions.

```bash
uv run jupyter lab notebooks
```

## Task 2: Databases

```bash
docker compose up -d
```

```bash
uv run python -m formative_1.load_to_db
```

The schema is `src/formative_1/tables.sql`. Set `MONGO_URL` if MongoDB is not
on `mongodb://localhost:27017`.

## Task 3: API

```bash
uv run fastapi dev src/formative_1/api.py
```

GET `/latest`, GET `/?start=&end=`, POST `/`, PUT and DELETE
`/{ticker}/{day}`. Docs at <http://127.0.0.1:8000/docs>.

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
