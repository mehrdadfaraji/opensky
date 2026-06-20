# Flight Phase Classifier

Month 1 project — ML + Autonomous Aviation Journey

## What it does

Classifies live flights over Canada into one of three phases — **climb**, **cruise**, or **descent** — using real-time ADS-B telemetry from the OpenSky Network.

## Why it's interesting

Most ML classifiers use pre-labeled datasets. Here, the labels are derived from physics: a plane climbing has a positive vertical rate, descending has a negative one. But the model itself predicts phase from **altitude and speed alone** — without seeing vertical rate — which means it's learning the relationship between where a plane is and how fast it's going, not just reading off a threshold.

## Data

Live flight data fetched from the [OpenSky Network REST API](https://opensky-network.org) — no authentication required.

- ~1400 flights over Canada per snapshot
- Features used: `baro_altitude` (meters), `velocity` (m/s)
- Labels derived from `vertical_rate`:
  - `climb` → vertical rate > 2 m/s
  - `descent` → vertical rate < -2 m/s
  - `cruise` → everything else

## Model

Random Forest Classifier wrapped in a sklearn Pipeline with a mean imputer for missing values.

**Cross-validation accuracy (5-fold): 72.6%**

The ceiling here is intentionally limited — two features, live snapshot data, no temporal context. Good enough for a first ship.

## How to run

```bash
conda activate ml_env
jupyter lab
```

Open `01_opensky_classifier.ipynb` and run all cells. Fetches live data on every run.

## What I learned

- Deriving labels from domain knowledge instead of manual annotation
- Handling missing ADS-B values with imputation instead of dropping rows
- Why 100% accuracy is a red flag (data leakage from `vertical_rate`)
- Building a reusable sklearn Pipeline for future projects
