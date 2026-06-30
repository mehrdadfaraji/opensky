# METAR Weather Condition Classifier

Month 2 project — ML + Autonomous Aviation Journey

## What it does

Classifies current aviation weather conditions at Canadian airports as **VFR** or **non-VFR** using live METAR data from the Aviation Weather Center API. A second notebook loads the exported model and validates predictions against real-time observations.

## Why it matters

As a pilot, the VFR/IFR distinction is the most important weather decision before every flight. This project automates that classification from raw telemetry — no manual weather briefing needed.

## Data

Live METAR data from the [Aviation Weather Center API](https://aviationweather.gov/api/data/metar) — no authentication required.

- 8 major Canadian airports: CYYZ, CYVR, CYUL, CYEG, CYYC, CYOW, CYWG, CYQB
- 24 hours of observations per fetch (~230 rows)

## Feature engineering

Raw METAR fields require extraction and transformation:

| Feature | Source | Notes |
|---------|--------|-------|
| `visib` | API field | Strip `+` suffix, convert to float |
| `wspd` | API field | Wind speed in knots |
| `ceiling` | `clouds` list | Lowest BKN or OVC layer in feet — None if clear |
| `temp_dewp_spread` | `temp - dewp` | Small spread = fog/low cloud risk |

**Labels** derived from `fltCat` field (pre-computed by the API using FAA rules):
- VFR → ceiling >3000ft AND visibility >5sm
- MVFR / IFR / LIFR → grouped as `non-VFR`

## Model

Random Forest Classifier wrapped in a sklearn Pipeline with mean imputation for missing ceiling values (clear sky = no ceiling layer).

**Cross-validation accuracy (5-fold): 98.7%**

Near-perfect separation is expected — ceiling and visibility are the exact inputs the FAA uses to define these categories. The model learns the boundary, not a proxy for it.

## Notebooks

- `vfr_predictor.ipynb` — fetches data, engineers features, trains and exports the model
- `metar_model_predictor.ipynb` — loads the saved model, fetches fresh data, compares predictions to real values

## How to run

```bash
conda activate ml_env
jupyter lab
```

Run `vfr_predictor.ipynb` first to train and save the model, then `metar_model_predictor.ipynb` to validate on fresh data.

## What I learned

- Fetching and parsing structured aviation weather data
- Extracting ceiling from nested cloud layer lists
- Feature engineering from domain knowledge (temp/dewpoint spread, ceiling extraction)
- Exporting and reusing a trained sklearn Pipeline with `joblib`
- Why class imbalance matters — and when near-perfect accuracy is legitimate vs suspicious
