# Month 4 — Sepsis Early-Warning Detector

**Sector:** Health (pivoted from aviation per the technique-first, sector-per-project principle — see `ML_AVIATION_JOURNEY.md`)
**Technique:** Time series, sliding windows, rare-event classification
**Dataset:** [PhysioNet 2019 Sepsis Prediction Challenge](https://physionet.org/content/challenge-2019/1.0.0/) — 40,336 ICU patients across two hospital systems (`training_setA`, `training_setB`), hourly vitals + labs, ~2-4% positive rate depending on split.
**Notebook:** `sepsis-prediction.ipynb`

---

## The task

Predict, hour by hour, whether an ICU patient is going into sepsis — using only data available up to that hour. This is the Challenge's own framing: `SepsisLabel` is pre-shifted to represent early warning (positive starting up to 6 hours before clinical onset), so predicting the label at the current hour using a window of recent history *is* the early-warning task, not a simplification of it.

## Why set A → train, set B → test

`training_setA` and `training_setB` come from two different hospital systems (confirmed in the official Challenge paper). Using A for training/validation and B as a fully held-out test set gives two things a random row-level split can't:

- **Zero leakage risk** — no patient can possibly appear in both, since they're from different hospitals entirely.
- **A genuine cross-hospital generalization test** — does the model work on a hospital it's never seen, not just on patients it hasn't seen from the same hospital.

## Pipeline

1. **Per-patient cleaning** (`clean_patient`) — drop the one 100%-missing column (`EtCO2`), forward-fill within each patient, fall back to a population median (computed from `training_setA` only, never from the test hospital) for any leading gap forward-fill can't reach.
2. **Sparse-lab flags** — six labs (`TroponinI`, `Bilirubin_direct`, `Fibrinogen`, `Bilirubin_total`, `Alkalinephos`, `AST`) are missing 98.5-99.9% of the time, but *whether* they were ever measured is itself a signal (clinicians order them when they already suspect a problem) — captured as `<lab>_ever_measured` flags before filling.
3. **Sliding windows** (`build_windows`) — 6-hour rolling windows. Vitals + labs get `mean/min/max/trend` (mean for typical level, min/max for extremes a mean would hide, trend for direction of change); static facts, flags, and `hours_since_lactate` are passed through as-is rather than meaninglessly aggregated.
4. **Patient-level splitting** — `GroupShuffleSplit` on `patient_id` for the train/validation split within set A, so no patient's windows cross between train and validation either.

## Model selection

Random Forest was tried first (Jeremy Howard's "always try RF first for tabular" — same playbook as month 3), both default and `class_weight='balanced'`. Both effectively found nothing at the default 0.5 threshold — recall of 0.4% and 0.1% respectively, confirmed stable (not a single-split fluke) across a 5-seed rigor check.

The real story only emerged from a PR-AUC / threshold sweep: **both RF variants had genuine ranking ability well above random (3.4-3.8x), it was just invisible at the default threshold** given how rare the positive class is. Gradient boosting (`HistGradientBoostingClassifier`, with balanced sample weights) did better still — PR-AUC 4.5x random on validation, and consistently better precision than RF at every matched recall level. GBM was selected as the final model.

**Sanity check against a real benchmark:** the official Challenge used a custom utility score, not PR-AUC, so scores aren't directly comparable — but one published competition entry using plain Random Forest reported AUPRC 0.045 on their held-out Test Set A. This project's numbers (validation PR-AUC 0.1005, test PR-AUC 0.0659) landed above that reference point despite using simpler features and no ensembling of model types.

## Results

Operating threshold (0.7648) chosen from the validation PR curve to target recall ≈ 30% / precision ≈ 12.5% — a deliberate trade-off favoring catching more true cases over fewer false alarms, matching how the Challenge's own utility function penalizes missed detections far more than false positives.

| | Validation (set A, held-out patients) | Test (set B, held-out hospital) |
|---|---|---|
| Positive rate | 2.23% | 1.43% |
| Recall | 0.300 | 0.190 |
| Precision | 0.125 | 0.100 |
| PR-AUC | 0.1005 | 0.0659 |
| PR-AUC vs. random baseline | ≈4.5x | ≈4.6x |

**The headline finding: the model's relative ranking ability transferred cleanly across hospitals (≈4.5x random on both), even though the fixed probability threshold didn't** — recall dropped from 30% to 19% and precision from 12.5% to 10% on the new hospital. This is a real, expected effect: a threshold calibrated on one hospital's patient population and prevalence doesn't map perfectly onto a different hospital's, even when the underlying model generalizes well. In a real deployment, this points to recalibrating the threshold per-site rather than distrusting the model itself.

## What didn't work, and why (worth keeping for the next session)

- **Window aggregation (`mean/min/max/trend`) is close to meaningless for high-missingness labs**, because forward-fill happens before windowing — a lab measured once and carried forward produces a window of six identical values, so `min=max` and `trend=0` almost always. This wasn't fixed this month (the GBM result made a feature rebuild unnecessary for now) but is the clear next lever if performance needs to improve: replace lab window-stats with recency-based features (`hours_since_X`, extended from `Lactate` to all six sparse labs).
- **`class_weight='balanced'` made Random Forest worse, not better** — confirmed stable across 5 seeds, the opposite of what happened in month 3's NTSB project. A reminder that "balanced always helps" isn't a universal rule.
- Dropping `ICULOS` (hours since ICU admission) collapsed RF to zero true positives — the model's tiny early signal was riding heavily on a temporal proxy rather than genuine physiological deterioration patterns, until GBM found more distributed signal across the feature set.

## Tools

First month using MLflow for experiment tracking from the start (`run_rf_experiment` logs params/metrics per run) — see `ML_AVIATION_JOURNEY.md`'s running-practices note.
