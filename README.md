# ML → Autonomous Aviation Journey

A 24-month roadmap from software engineering into ML and autonomous aviation systems, one project per month, every project aviation-flavored. Full plan and running log: see `ML_AVIATION_JOURNEY.md` in the parent journal (not part of this repo).

## Projects

| month | project | techniques | status |
|---|---|---|---|
| [1](month-01-flight-phases/) | Flight phase classifier — climbing / cruising / descending from live ADS-B data | Random forest, sklearn Pipeline, label derivation from physics | 72.6% CV accuracy |
| [2](month-02-metar-classifier/) | METAR weather classifier — VFR vs non-VFR from live weather data | Random forest, feature engineering from raw METAR fields | 98.7% CV accuracy |
| [3](month-03-ntsb-accidents/) | GA accident fatality prediction — three-stage project on the NTSB's full historical accident database | From-scratch NN (PyTorch), BCE vs L1 loss, seed-collapse rigor checks, random forests, feature importance | up to 79% recall (tuned RF), see project README for the honest tradeoffs |

Each project folder has its own README with data sources, approach, results, and (for month 3) a plain-language write-up alongside the technical one.

## Setup

```bash
conda env create -f environment.yml   # or the project-specific environment.yml inside each folder
conda activate ml_env
jupyter lab
```

## Why this structure

Sequential, not simultaneous — one project at a time, each one shipped (working code + README + a plain-language explanation of what it does and why) before moving to the next. Months 1-2 are self-contained single notebooks; month 3 grew into a three-stage investigation (a documented failure, a fix with a rigor check, and a from-scratch-vs-random-forest comparison) because the first honest result raised real follow-up questions worth chasing before moving on.
