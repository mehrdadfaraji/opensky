"""
Module A, step 2: turn keypoints.parquet (long format, one row per landmark
per frame) into a fixed-size feature table (one row per clip) that a
classifier can actually use.

Design (agreed on in conversation, not arbitrary):
- Phase-based aggregation: each clip's frames are split into 3 equal phases
  (early / mid / late) and each raw signal is averaged within each phase.
  This keeps some of the temporal shape (unlike a single whole-clip average)
  without needing a full sequence model.
- Delta features (late - early) are added on top for the signals most
  likely to trend meaningfully across a swing (elbow angle, wrist height,
  wrist speed) — cheap for a tree-based model to use directly, instead of
  making it reconstruct the trend from three separate columns.
- Low-visibility landmarks (< --min-visibility) are treated as missing for
  any signal that depends on them, not blindly trusted.
- Wrist speed is normalized by shoulder width (a rough body-scale reference)
  so it's more comparable across subjects at different distances from the
  camera. It's expressed as displacement per *frame*, not per second —
  we don't have per-clip fps saved in keypoints.parquet, and THETIS clips
  come from one consistent Kinect capture setup, so frame-based is a
  reasonable simplification. Worth revisiting if results look fps-sensitive.

Usage:
    python feature_engineering.py --keypoints keypoints.parquet --out features.parquet
"""

import argparse
import math

import numpy as np
import pandas as pd

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

N_PHASES = 3


def _get(frame, landmark_id, min_visibility):
    """Return (x, y) for a landmark in a frame dict, or None if missing/low-visibility."""
    row = frame.get(landmark_id)
    if row is None or row["visibility"] < min_visibility:
        return None
    return row["x"], row["y"]


def _angle(a, b, c):
    """Angle at point b, given three (x, y) points. None if any point missing."""
    if a is None or b is None or c is None:
        return np.nan
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return np.nan
    cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # clamp floating-point drift
    return math.degrees(math.acos(cos_angle))


def _dist(a, b):
    if a is None or b is None:
        return np.nan
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clip_to_frames(clip_df, min_visibility):
    """Long-format rows for one clip -> {frame_idx: {landmark_id: {x,y,visibility}}}, sorted."""
    frames = {}
    for frame_idx, group in clip_df.groupby("frame_idx"):
        frames[frame_idx] = {
            int(r.landmark_id): {"x": r.x, "y": r.y, "visibility": r.visibility}
            for r in group.itertuples()
        }
    return dict(sorted(frames.items()))


def compute_raw_signals(frames, min_visibility):
    """
    Per-frame body signals for one clip, as a DataFrame indexed by frame order
    (not raw frame_idx, since some frames may be missing entirely).
    """
    records = []
    prev_wrist = {"left": None, "right": None}
    prev_frame_idx = None

    for frame_idx, lm in frames.items():
        ls = _get(lm, LEFT_SHOULDER, min_visibility)
        rs = _get(lm, RIGHT_SHOULDER, min_visibility)
        le = _get(lm, LEFT_ELBOW, min_visibility)
        re = _get(lm, RIGHT_ELBOW, min_visibility)
        lw = _get(lm, LEFT_WRIST, min_visibility)
        rw = _get(lm, RIGHT_WRIST, min_visibility)
        lh = _get(lm, LEFT_HIP, min_visibility)
        rh = _get(lm, RIGHT_HIP, min_visibility)
        lk = _get(lm, LEFT_KNEE, min_visibility)
        rk = _get(lm, RIGHT_KNEE, min_visibility)
        la = _get(lm, LEFT_ANKLE, min_visibility)
        ra = _get(lm, RIGHT_ANKLE, min_visibility)

        shoulder_width = _dist(ls, rs)

        def speed(prev, cur):
            if prev is None or cur is None or not shoulder_width or np.isnan(shoulder_width):
                return np.nan
            gap = frame_idx - prev_frame_idx if prev_frame_idx is not None else 1
            gap = max(gap, 1)
            return _dist(prev, cur) / gap / shoulder_width

        rec = {
            "frame_idx": frame_idx,
            "left_elbow_angle": _angle(ls, le, lw),
            "right_elbow_angle": _angle(rs, re, rw),
            "left_knee_angle": _angle(lh, lk, la),
            "right_knee_angle": _angle(rh, rk, ra),
            "left_wrist_rel_height": (ls[1] - lw[1]) if ls and lw else np.nan,
            "right_wrist_rel_height": (rs[1] - rw[1]) if rs and rw else np.nan,
            "torso_rotation_deg": (
                math.degrees(math.atan2(rs[1] - ls[1], rs[0] - ls[0])) if ls and rs else np.nan
            ),
            "left_wrist_speed": speed(prev_wrist["left"], lw),
            "right_wrist_speed": speed(prev_wrist["right"], rw),
        }
        records.append(rec)
        prev_wrist = {"left": lw, "right": rw}
        prev_frame_idx = frame_idx

    return pd.DataFrame(records)


SIGNAL_COLUMNS = [
    "left_elbow_angle", "right_elbow_angle",
    "left_knee_angle", "right_knee_angle",
    "left_wrist_rel_height", "right_wrist_rel_height",
    "torso_rotation_deg",
    "left_wrist_speed", "right_wrist_speed",
]

# Signals worth an explicit late-minus-early trend feature (see module docstring).
TREND_SIGNALS = [
    "left_elbow_angle", "right_elbow_angle",
    "left_wrist_rel_height", "right_wrist_rel_height",
    "left_wrist_speed", "right_wrist_speed",
]


def phase_aggregate(signals_df):
    """One clip's per-frame signals -> one row of phase-aggregated + trend features."""
    n = len(signals_df)
    if n == 0:
        return {}

    phase_edges = np.linspace(0, n, N_PHASES + 1, dtype=int)
    phase_names = ["early", "mid", "late"]
    out = {}

    phase_means = {}
    for col in SIGNAL_COLUMNS:
        means = []
        for i, name in enumerate(phase_names):
            chunk = signals_df[col].iloc[phase_edges[i]:phase_edges[i + 1]]
            m = chunk.mean(skipna=True)
            out[f"{col}__{name}"] = m
            means.append(m)
        phase_means[col] = means

    for col in TREND_SIGNALS:
        early, _, late = phase_means[col]
        out[f"{col}__trend"] = (
            late - early if not (np.isnan(early) or np.isnan(late)) else np.nan
        )

    return out


def build_feature_table(keypoints_df, min_visibility):
    rows = []
    meta_cols = keypoints_df[["clip_id", "actor", "action", "seq"]].drop_duplicates("clip_id")
    meta_lookup = meta_cols.set_index("clip_id").to_dict("index")

    for clip_id, clip_df in keypoints_df.groupby("clip_id"):
        frames = clip_to_frames(clip_df, min_visibility)
        signals_df = compute_raw_signals(frames, min_visibility)
        features = phase_aggregate(signals_df)
        meta = meta_lookup[clip_id]
        rows.append({"clip_id": clip_id, **meta, "n_frames": len(frames), **features})

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keypoints", default="keypoints.parquet")
    ap.add_argument("--out", default="features.parquet")
    ap.add_argument("--min-visibility", type=float, default=0.5)
    args = ap.parse_args()

    df = pd.read_parquet(args.keypoints)
    print(f"Loaded {len(df)} keypoint rows across {df['clip_id'].nunique()} clips.")

    feature_df = build_feature_table(df, args.min_visibility)

    n_missing = feature_df.isna().any(axis=1).sum()
    print(f"Built {len(feature_df)} clip rows, {feature_df.shape[1]} columns "
          f"({n_missing} rows have at least one NaN feature — expected for clips "
          f"with detection gaps, handled at training time).")

    feature_df.to_parquet(args.out, index=False)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
