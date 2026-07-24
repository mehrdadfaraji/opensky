# Module A — Tennis Stroke Classifier (CV / pose keypoints)

EA Interview Sprint, Module A. Pose-keypoint-based stroke classification on the THETIS dataset — pulls Phase 2's CV/CNN work forward, maps to EA's "preferred: Computer Vision (for video)" requirement.

## Engineering Skill Targets
*(see ML_AVIATION_JOURNEY.md → Running Practices → README skill-check)*

1. **Problem framing** — input: MediaPipe pose keypoints extracted from THETIS RGB video clips. Target: which of 4 strokes (`backhand`, `flat_service`, `forehand_flat`, `smash`). Baseline: logistic regression floor.
2. **Data understanding** — raw video properties inspected; keypoints visually confirmed by overlaying them back on frames (`visualize_clip.py`); the visibility field and its two kinds of missingness (occlusion vs. out-of-frame) understood before trusting any feature built on top of it.
3. **Evaluation rigor** — subject-based (`actor`) group split, not clip-based — a subject's clips can never appear in both train and test. Evaluated across 5 different random group-splits for the tabular models, 3 seeds for the CNN (same rigor habit as NTSB's 5-splits × 5-seeds table).
4. **Under-the-hood understanding** — MediaPipe's pose model: used, not built (pretrained, frozen, inference-only — that's the honest boundary). The feature engineering, all three tabular classifiers, and the 1D-CNN (architecture, forward pass, loss, backprop, all explicit): built here.
5. **Build vs. use** — off-the-shelf: MediaPipe pose detection. Built here: phase-aggregated features + trend deltas, the tabular model comparison, and the 1D-CNN's resampling/architecture/training loop.
6. **Integration** — one step in a pipeline: raw video → `extract_keypoints.py` (batch script) → `feature_engineering.py` / this analysis (features + models) → later, Module D wraps the winning model in a coaching agent.

---

## What this project does

Classifies a tennis stroke from a short video clip into one of 4 classes — `backhand`, `flat_service`, `forehand_flat`, `smash` — using only 2D body pose keypoints, not raw pixels. Two approaches are built and compared: tabular models on hand-engineered phase-aggregated features, and a 1D-CNN on the raw per-frame signal sequence.

## Data

**THETIS** (github.com/THETIS-dataset/dataset) — real academic action-recognition dataset, 55 subjects, RGB video + Kinect skeleton data. Only 4 of the dataset's 12 stroke classes were pulled (`download_thetis.sh`): the 4 biomechanically distinct stroke families, not the harder within-family sub-variants (`forehand_slice`, `backhand2hands`, `kick_service`, etc.). Deliberate staged scope — get the full pipeline (extract → features → classify → evaluate) working end-to-end on a clean 4-class problem first; expanding to the remaining 8 classes is a documented follow-up, not a cut corner.

- 660 clips, all 55 subjects, roughly balanced across the 4 classes (384–444 clips each).
- Pose extraction: MediaPipe Tasks API (`PoseLandmarker`, `extract_keypoints.py`) — the legacy `mp.solutions.pose` API is dead in current MediaPipe releases (see Known Issues in the main journal), so this project already had to route around that.
- Output: `keypoints.parquet`, long format — one row per landmark per frame per clip (1.65M rows), `x`/`y`/`z`/`visibility` per landmark.
- Missingness: landmarks with `visibility < 0.5` are treated as missing, not trusted, for any feature that depends on them — confirmed necessary by visually overlaying keypoints on frames and seeing real detection dropout during fast motion (the swing itself).

## Approach

**Two parallel model families, same data, same evaluation protocol, deliberately compared:**

**1. Tabular models on phase-aggregated features** (`feature_engineering.py`, `tennis-pose-classifier.ipynb`)
Each clip is split into 3 equal phases (early/mid/late); 9 raw per-frame signals (elbow/knee angles, wrist height relative to hip, torso rotation, wrist speed normalized by shoulder width) are averaged within each phase, plus explicit early→late trend deltas for 6 of them — 33 features per clip. Rationale: a whole-clip average would erase the swing's temporal shape entirely, but a full sequence model is more than needed as a first pass; phase aggregation is the middle ground, and handing trees an explicit trend feature saves them from reconstructing "is late bigger than early" across several splits with only ~660 clips to learn from.

Three-tier comparison, same escalation pattern as the NTSB project: logistic regression (floor) → random forest (solid default) → XGBoost (modern tabular standard).

**2. 1D-CNN on the raw sequence** (`pose_tennis_cnn.ipynb`, separate notebook/kernel — `xgboost`'s Homebrew `libomp` and PyTorch's bundled OpenMP deadlock silently in the same process on macOS, see Known Issues)
Same 9 per-frame signals, *not* phase-aggregated — resampled to a fixed 64 timesteps per clip (linear interpolation) so clips of different lengths are directly comparable. The architecture deliberately avoids global pooling at the end: it flattens the conv output while keeping the time axis, so *where* along the clip a pattern fired isn't thrown away. A global-pooled CNN can't represent "toss happens before the swing" vs. "no toss" — this one can. This is the one model in the whole project trained by hand end-to-end: architecture, forward pass, loss, backprop all explicit in the notebook, not a black-box `.fit()` call.

Both use the same subject-based group split and per-channel standardization fit on train data only (no test-set leakage).

## Results

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Logistic regression | 0.663 ± 0.041 | 0.662 ± 0.039 |
| Random forest | 0.662 ± 0.048 | 0.661 ± 0.046 |
| XGBoost | 0.671 ± 0.023 | 0.670 ± 0.020 |
| **1D-CNN (raw sequence)** | **0.742 ± 0.022** | **0.741 ± 0.025** |

(Tabular: 5 group-splits. CNN: 3 seeds. 4-class problem, so chance is 25%.)

The 1D-CNN beats every tabular model by ~7-8 points and is also the most consistent (tightest spread). The tabular models' confusion matrix showed a specific, repeated failure: `smash` and `flat_service` get confused with each other (16 smash→flat_service, 11 flat_service→smash in one split) — biomechanically sensible, since a smash is essentially a serve motion hit from a different court position, and phase-averaging can blur exactly the timing cue (when the toss happens relative to the swing) that tells them apart. The CNN's full-sequence view directly addresses this: smash precision rose from 0.57 to 0.80, and `smash`/`flat_service` recall both improved in the CNN's classification report vs. random forest's. This is the concrete payoff of not collapsing the time axis away.

Feature importance (random forest / XGBoost, tabular models): `right_wrist_rel_height__mid`, `left_elbow_angle__early`/`__late`, and `left_wrist_rel_height__late`/`__trend` dominate — consistent with the biomechanical intuition that stroke identity is mostly readable from arm geometry, and that *when* in the swing you look (early vs. late) matters more than any single instantaneous reading.

## What I'd do next

- Expand to the remaining 8 THETIS classes (`forehand_slice`, `backhand2hands`, `kick_service`, etc.) — `download_thetis.sh` already documents the `git sparse-checkout add` command for this; the pipeline doesn't need architectural changes, just more data and (likely) a small CNN capacity bump.
- A brief handedness/saliency investigation into the CNN's remaining smash↔flat_service misclassifications was started (using per-clip saliency maps and a YOLO-based racket-hand detector) but didn't reach a clean conclusion on the small sample of misclassified clips and was dropped rather than chased further — noted here in case it's worth revisiting with more data.
- k-fold CV instead of a single held-out split per seed (same open item as NTSB).
- Feed the CNN's per-clip features into Module D's coaching agent as the "what stroke is this" tool call.

## Non-technical explanation
*(per the monthly completion rule — explain what it does and why it works to someone with no ML background)*

Take a video clip of someone hitting a tennis ball. First, a pretrained pose-detection model (like the skeleton overlay you've seen in sports broadcasts) tracks where the player's joints — shoulders, elbows, wrists, knees — are in every frame. That turns a video into a much simpler set of numbers: joint positions over time, instead of millions of pixels.

From there, two different strategies were tried to answer "which of 4 strokes is this?" The first summarizes the swing into a handful of numbers (how bent was the elbow near the start vs. the end, how fast did the wrist move, etc.) and feeds those to fairly standard, well-understood models. That got it right about 2 times out of 3. The second strategy keeps the full motion in order — the actual timeline of the swing, frame by frame — and lets a more specialized model (built from scratch here, not off-the-shelf) learn directly from that sequence. That got it right about 3 times out of 4, and specifically fixed the mistake the first approach kept making: mixing up a serve with a smash, two strokes that look almost identical in a single freeze-frame but unfold differently over time (a smash has a ball already in the air; a serve has a toss first). The lesson: for anything where *order and timing* matter, showing the model the whole sequence beats summarizing it away — even when the summarized version is far cheaper to build.
