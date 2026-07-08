# GA Accident Fatality Prediction — Decision Tree / Random Forest

**File:** `ntsb-accident-random-forest.ipynb`

**Goal:** same prediction target as the rest of this project — will a general-aviation accident
be fatal — but a different modeling approach, following jhoward's
[How Random Forests Really Work](https://www.kaggle.com/code/jhoward/how-random-forests-really-work)
instead of building a neural net from scratch. Motivation: the from-scratch linear model / NN in
`ntsb-accident-pred.ipynb` / `ntsb-accident-pred-extended.ipynb` needed real care to find a weak
signal (right loss function, right depth, right init scale, a real seed-collapse rate). For tabular
data specifically, the standard advice is to reach for a random forest first — it needs almost no
tuning, doesn't care about feature scaling or encoding scheme, and hands back a ranked feature
importance list for free. This notebook tests that claim on real data.

## Data

Same `events.csv` (NTSB CAROL export), but a leaner, hand-picked feature set than the extended
notebook's — 28 features (9 categorical, 19 continuous): weather at the time of the accident,
location relative to the nearest airport, and basic timing. No pilot/aircraft table joins in this
notebook (unlike the extended notebook, where `flight_hours` turned out to be the single strongest
predictor available).

Categorical columns (`ev_dow`, `ev_nr_apt_loc`, `ev_state`, `light_cond`, `mid_air`,
`on_ground_collision`, `sky_cond_ceil`, `sky_cond_nonceil`, `wx_cond_basic`) are converted to
integer codes via pandas `.cat.codes`, not one-hot encoded — trees don't need dummy variables the
way linear models do, since a split only ever asks "is this value above or below a threshold,"
never "how much does this contribute to a weighted sum."

## Approach

1. Drop leakage columns (injury/fatality counts — literally what the target is derived from — and
   `ev_type`, which turned out to be leakage too: it's defined by injury/damage severity thresholds
   per 49 CFR 830.2, 22% fatal for "accidents" vs 0.4% for "incidents").
2. Drop admin/ID/near-empty/redundant columns (65 → 29 columns).
3. Fit a tiny 4-leaf tree first, purely to see the mechanics — verified its gini-impurity values by
   hand to confirm understanding of what the algorithm is actually doing at each split.
4. Sweep `min_samples_leaf` on a single tree (100 → 10) to find the point where validation
   performance stops improving — the real stopping criterion, not a guessed number.
5. Move to `RandomForestClassifier`, same sweep, plus `class_weight='balanced'` (needed — the
   default forest is still biased toward the majority "non-fatal" class, same issue as a single
   estimator).
6. Pick a final model, check its feature importances.

## Key findings

- **`ev_nr_apt_loc` (on-airport vs. off-airport) is the single strongest predictor**, by more than
  3x the next feature. On-airport accidents: ~7% fatal. Off-airport or unrecorded location: ~30%
  fatal — roughly 4x higher, and it matches real aviation-safety intuition (on-airport mishaps tend
  to be lower-severity; off-airport accidents are more often genuine in-flight emergencies).
- Missing values in `ev_nr_apt_loc` behave almost identically to genuinely off-airport ones
  (29.5% vs. 30.8% fatal) — confirmed this rather than assumed it, since the tree's default handling
  of missing categorical values (a separate `-1` code) happened to group them together.
- **A default random forest actually has *worse* recall than a well-tuned single tree** (0.23 vs.
  0.32) — averaging many trees doesn't fix a class-imbalance bias baked into every one of them, it
  just makes that bias more stable and confident. `class_weight='balanced'` is what actually moves
  the needle.
- The `min_samples_leaf` sweep found a genuine recall ceiling around **0.75–0.8** for this feature
  set: pushing the forest to grow deeper (`leaf=5`) inflates validation recall through overfitting
  (train recall 0.91 vs. val recall 0.46, a wide gap); backing off to `leaf=100` gets a much smaller
  train/val gap at effectively the same validation recall — a more trustworthy version of the same
  result, not a worse one.

## Final model and results

`RandomForestClassifier(100, min_samples_leaf=100, class_weight='balanced')`

| model | val recall | val precision |
|---|---|---|
| single tree, `min_samples_leaf=25` | 0.32 | 0.50 |
| RF, default (unbalanced) | 0.23 | 0.68 |
| **RF, balanced, `min_samples_leaf=100` (final)** | **0.79** | 0.34 |

Catches roughly 1 in 4 real fatal accidents with the unbalanced default, versus roughly 4 in 5 with
the tuned, balanced forest — at real cost to precision (more false alarms). Not a free win: this is
a different point on the same recall/precision tradeoff every model in this project has run into,
not a strictly better model.

## In plain terms

Imagine sorting thousands of past accidents into groups by asking yes/no questions — "did this
happen on airport property?", "was the weather clear?" — and seeing what fraction of each resulting
group turned out fatal. That's a decision tree. A random forest builds a hundred slightly different
versions of that sorting process (each one looking at a random subset of the data and questions) and
averages their opinions, which smooths out the mistakes any single tree would make on its own.

The one dial that mattered most here is telling the forest "the rare, important outcome (fatal)
matters as much as the common one (non-fatal), even though it's only 1 in 5 accidents" — without
that, the model quietly defaults to assuming nothing is ever fatal, since guessing "not fatal" is
right 4 times out of 5 for free. Once that's fixed, the single most useful piece of information
this model found, out of 28 factors it was given, was simply whether the accident happened on
airport property or somewhere else — a bigger factor than any specific weather reading.

## Not yet done

- Precision-recall threshold tuning instead of sklearn's fixed 0.5 cutoff.
- Proper k-fold cross-validation instead of a single 75/25 split.
- Joining the pilot/aircraft tables the extended notebook used — `flight_hours` was that notebook's
  top feature and isn't included here at all; likely real signal left on the table.
- Gradient-boosted trees (XGBoost/LightGBM), the natural next rung up from random forests for
  tabular problems.
