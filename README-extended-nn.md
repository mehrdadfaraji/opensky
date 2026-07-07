# GA Accident Fatality Prediction — Extended NN + Trustworthiness Investigation

**File:** `ntsb-accident-pred-extended.ipynb`

**Goal:** same target as the rest of this project — will a GA accident be fatal — starting from the
working (but weak) BCE + shallow-net fix from `README-linear-model.md`, then asking the obvious
follow-up: is that 13% recall number solid, and is weather really the best signal available?

## Data

`events.csv` (NTSB CAROL export, 30,804 rows), same source as the linear-model notebook, later
joined against two more tables from the full `avall.mdb` Access database — `aircraft` and
`Flight_Crew`/`flight_time` — via `access_parser` (pure-Python, no `mdbtools`/root required).

Outlier handling on the base weather columns, done with a physically-grounded cutoff for each,
not a statistical rule of thumb: `vis_sm` capped at 15sm (real surface visibility never exceeds
this — everything above is a data-entry sentinel), `sky_ceil_ht` capped at 50,000ft (only 2 rows
above it, both impossible), `gust_kts` drops one `999.0` sentinel row, `wx_temp` capped at 140°F
(hottest ambient temperature ever recorded on Earth is ~134°F). 30,804 → 30,292 rows survive.
`wind_vel_kts` deliberately left untouched — no sentinel pattern, and severe wind is a real causal
factor in accident data, not noise.

## Approach

1. **Baseline (7 features):** ceiling, visibility, wind, gust, temperature, VMC/IMC one-hot.
   Single hidden layer (20 units), BCE loss, centered init `(rand-0.5)*0.1` — the fix carried over
   from the linear-model notebook. Result: ~13% recall, ~53% precision, ~79.6% accuracy.
2. **Rigor check:** re-run the same model across 5 split seeds × 5 init seeds (25 runs). Result:
   5/25 (20%) collapse to 0% recall (stuck predicting the base rate); the other 20 land around
   13-14%. Conclusion: the number is real but not guaranteed — a single training run can silently
   report a useless model that still looks like ~80% accuracy.
3. **Untapped signal already in `events.csv`:** only 6 of 73 available columns were used. Checked
   `light_cond` and `mid_air` against fatal rate directly — `light_cond=NDRK` (dark, unlit) is
   51.1% fatal vs. 16.3% for daylight, a bigger swing than any weather feature. Added
   `light_cond`, `mid_air`, cyclical month (sin/cos), and airport distance — all free, already in
   the CSV. Result: recall ~13-14% (flat), precision up to ~62%, accuracy 80.4%, **0/25 seed
   collapses** — richer features made training itself reliable, not just more accurate.
4. **Joining `avall.mdb`'s other tables:** pilot age, medical certificate, logged flight hours,
   aircraft type, FAR part — none of which reach `events.csv`. Found and fixed a real bug on the
   way: `merge()` resets the DataFrame index, so `df.loc[X.index, ...]` after a merge silently
   pulls the *wrong rows'* labels. Every seed collapsed to 0% recall regardless of optimizer or
   learning rate until this was fixed (look labels up by `ev_id` instead — the one column `merge`
   preserves). Also handled two data-quality issues: negative ages and a `999999` sentinel for
   flight hours, both treated as missing rather than real values.

## Key findings

| feature set | recall | precision | accuracy | seed collapse |
|---|---|---|---|---|
| 7 weather features (baseline) | 11-14% | 57% | 80.0% | 5/25 |
| + `light_cond`, `mid_air`, season, airport distance | 13-14% | 62% | 80.4% | 0/25 |
| + pilot age/hours, aircraft type, `far_part` | 17-22% | 67-70% | 81-82% | 0/25 |

Recall roughly doubled and precision rose 10-13 points over the original — using only data
already free and available in the same historical database (30,292 rows is the full dataset, not
a sample; there's no "more data" to collect, only more *signal* to mine from what already exists).

A quick random-forest pass on the richest feature set (`X2`/`y2`, 36 columns) is also in this
notebook: default RF gets 25.2% recall / 59.4% precision with zero tuning; `class_weight='balanced'`
pushes recall to 36-40% (0/5 seed collapses across splits) at 46-48% precision — beating the NN's
best recall for a fraction of the effort. `flight_hours` (pilot experience) ranks as the single
strongest predictor, ahead of every weather variable, with season (`month_sin`/`month_cos`)
showing up as a new signal the NN's opaque weights never surfaced. See `README-random-forest.md`
for a deeper, separately-tuned pass on trees (different feature set, no table joins there).

## In plain terms

The first working model only looked at weather — clouds, wind, visibility — and found a weak
signal: roughly 1 in 8 fatal accidents caught. Before trusting that number, it got stress-tested
by retraining it 25 times with small random differences: 1 in 5 of those runs turned out
completely broken while looking fine on the surface, which is the kind of thing that's easy to
miss if you only ever train a model once.

Then the question became: is weather really the most anyone can learn from this data, or just
the easiest thing to reach for? It wasn't. Whether the accident happened in the dark, whether it
was a mid-air collision, and — once the extra effort of pulling in a second data table paid off —
how experienced the pilot was, all turned out to carry more signal than any weather reading. Pilot
flight hours ended up mattering more than the weather at the time of the crash. Digging into data
that was already free and already available, rather than assuming more of it was needed, roughly
doubled how many real fatal accidents the model could catch.

## Not yet done

- Precision-recall threshold tuning instead of a fixed 0.5 cutoff.
- Proper k-fold cross-validation instead of a fixed 80/25 split.
- `Findings`/`Occurrences` tables (probable cause codes) and phase-of-flight from narrative text.
- Gradient-boosted trees (XGBoost/LightGBM).
