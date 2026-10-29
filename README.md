# Forecast (Task 4)

`src/formative_1/forecast.py` fetches records from the API, applies the
training preprocessing, loads the LSTM ensemble from
`models/lstm_two_head.pt`, and predicts the next trading day's close.

## Run

```bash
git lfs pull
```

```bash
uv sync
```

```bash
uv run fastapi dev src/formative_1/api.py
```

In another terminal:

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
