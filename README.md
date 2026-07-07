# GA Accident Fatality Prediction

Month 3 project in a [24-month ML → autonomous aviation roadmap](../ML_AVIATION_JOURNEY.md).
**Goal:** predict whether a general-aviation accident will be fatal, using the NTSB's full public
accident history (CAROL database, `avall.mdb` / `events.csv`, 30,292 accidents after cleaning,
~21% fatal) — not a toy dataset.

This isn't one model, it's three attempts in increasing sophistication, each one a real lesson,
including the failures:

| stage | notebook | approach | best result |
|---|---|---|---|
| 1 | `ntsb-accident-pred.ipynb` | linear model, from scratch, L1 loss | **fails** — beaten by "always predict non-fatal" |
| 2 | `ntsb-accident-pred-extended.ipynb` | BCE + shallow net, then richer features + table joins | 17-22% recall, 67-70% precision, 81-82% accuracy |
| 3 | `ntsb-accident-random-forest.ipynb` | decision tree → tuned, balanced random forest | up to 79% recall, 34% precision (different feature set, tuned for recall) |

Each stage has its own README with the full technical writeup:
- [`README-linear-model.md`](README-linear-model.md) — why the first model failed, and why that's
  worth documenting rather than deleting.
- [`README-extended-nn.md`](README-extended-nn.md) — the fix (BCE, shallower net), a rigor check
  that caught an unstable result, and a feature/table-join hunt that roughly doubled recall.
- [`README-random-forest.md`](README-random-forest.md) — testing fast.ai's "always try a random
  forest first for tabular data" advice, with a `min_samples_leaf` sweep and feature importances.

A plain-language walkthrough of the whole arc, no ML background assumed, is in
[`NON_TECHNICAL_SUMMARY.md`](NON_TECHNICAL_SUMMARY.md).

## The honest headline

No version of this is a usable safety tool — catching somewhere between 1 in 5 and 4 in 5 fatal
accidents in advance (depending on which recall/precision tradeoff you pick) from
weather/pilot/aircraft metadata alone is a research result, not a product. What it demonstrates:
a from-scratch modeling pipeline surfacing real, reproducible, if weak, signal; a documented
architecture dead end (deeper isn't automatically better); a caught-and-fixed silent bug (index
misalignment after a `merge()`) that would otherwise have hidden a real improvement; and
confirmation, on real data, that random forests need far less tuning than a neural net for a
tabular problem like this one.

## Setup

```bash
conda activate ml_env
cd ntsb-carol-accidents
jupyter lab
```

`events.csv` and `avall.mdb` are gitignored (large data files) — download from
[data.ntsb.gov/avdata](https://data.ntsb.gov/avdata). See `README-extended-nn.md` for the
`mdbtools`/`access_parser` export step.

## Not yet done (project-wide)

- Precision-recall threshold tuning instead of sklearn's fixed 0.5 cutoff.
- Proper k-fold cross-validation instead of fixed train/val splits.
- Joining the pilot/aircraft tables into the random-forest notebook (stage 3 doesn't use
  `flight_hours` — the extended NN's top feature — at all yet).
- Gradient-boosted trees (XGBoost/LightGBM), the natural next rung up for tabular data.
- `Findings`/`Occurrences` tables (probable cause codes) and phase-of-flight from narrative text.
