# Bicing Availability Predictor

Predict how many bikes will be available at a Barcelona Bicing station in 15 minutes.

**[Live demo](https://huggingface.co/spaces/EricRistol/bicing-predictor)**

## Why predict bike availability

Bicing stations get empty or full unpredictably depending on time of day and neighborhood. Knowing availability 15 minutes ahead helps you plan your trip.

## The data

Generated synthetic snapshots matching the real Bicing GBFS API format. Three station types:
- residential: full at night, empties during the day
- central: empty at night, fills during work hours
- university: peaks around midday

Real data can be collected using `fetch_live.py` if you want to train on actual observations.

## Features used

- `hour`, `minute`, `dayofweek`, `is_weekend` -- time info
- `capacity`, `bikes_available`, `docks_available` -- current state
- `lag_1`, `lag_2`, `lag_3` -- availability 15/30/45 minutes ago
- `rolling_1h` -- mean bikes over the last hour
- station type (one-hot encoded)

Target: bikes_available at t+15 minutes.

## Models

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Just predict the current count | 2.44 | 3.12 | 0.79 |
| Linear Regression | 2.04 | 2.59 | 0.86 |
| Random Forest | 1.86 | 2.36 | 0.88 |
| LightGBM (best) | 1.84 | 2.33 | 0.89 |

LightGBM works best. Used MAE/RMSE/R² for evaluation.

## How to run

```bash
pip install -r requirements.txt
python main.py
```

Menu options:
- I: Generate dataset
- II: Explore the data
- III: Train and compare
- IV: Make a prediction
- V: Launch API
- VI: Quit

Or run scripts directly:
```bash
python train.py         # generates data, trains all models
python predict.py       # ask for station ID, print prediction
```

## Web API

```bash
python main.py          # pick option V
# or:
uvicorn api.app:app --reload
```

Open http://localhost:8000. Pick a station and see the 15-minute forecast.

---

**[Live demo](https://huggingface.co/spaces/EricRistol/bicing-predictor)**
