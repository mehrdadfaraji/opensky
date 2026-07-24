"""
Module A, step 1: turn THETIS RGB videos into a keypoints dataset.

This is the "call the API" part — MediaPipe does the pose detection, we're
just running it over every clip and saving the output in a usable shape.
The actual learning work starts after this: exploring what this data looks
like, engineering angle/velocity features, and building the classifier.

Uses MediaPipe's current Tasks API (PoseLandmarker) — the older
`mp.solutions.pose` API this script originally used was deprecated in 2023
and broke outright in recent mediapipe pip releases (0.10.31+, 2026).

Usage (inside your ml_env conda environment):
    pip install mediapipe opencv-python pandas tqdm pyarrow
    python extract_keypoints.py --video-dir data/thetis/VIDEO_RGB --out keypoints.parquet

Output: one long-format table with columns
    clip_id, actor, action, seq, frame_idx, landmark_id, x, y, z, visibility
33 landmarks per frame (MediaPipe's standard pose model: nose, shoulders,
elbows, wrists, hips, knees, ankles, etc. — the ones relevant to tennis are
roughly: 11/12 shoulders, 13/14 elbows, 15/16 wrists, 23/24 hips, 25/26 knees).
"""

import argparse
import re
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

FILENAME_RE = re.compile(r"(p\d+)_([a-zA-Z0-9]+)_s(\d+)\.avi")

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODEL_PATH = Path(__file__).parent / "pose_landmarker_lite.task"


def ensure_model():
    if not MODEL_PATH.exists():
        print(f"Downloading pose model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")
    return MODEL_PATH


def parse_filename(path: Path):
    """p10_smash_s1.avi -> actor='p10', seq=1 (action comes from the parent folder)."""
    m = FILENAME_RE.match(path.name)
    if not m:
        return None, None
    actor, _action_in_name, seq = m.groups()
    return actor, int(seq)


def extract_clip(video_path: Path, options, clip_id: str, actor: str, action: str, seq: int):
    rows = []
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # fallback if a file reports 0

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * (1000 / fps))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                # pose_landmarks is a list of detected poses; we expect one person per clip
                for lm_id, lm in enumerate(result.pose_landmarks[0]):
                    rows.append(
                        {
                            "clip_id": clip_id,
                            "actor": actor,
                            "action": action,
                            "seq": seq,
                            "frame_idx": frame_idx,
                            "landmark_id": lm_id,
                            "x": lm.x,
                            "y": lm.y,
                            "z": lm.z,
                            "visibility": lm.visibility,
                        }
                    )
            frame_idx += 1
    cap.release()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True, help="Path to THETIS VIDEO_RGB folder")
    ap.add_argument("--out", default="keypoints.parquet")
    ap.add_argument("--resume", action="store_true", help="Skip clips already in --out")
    args = ap.parse_args()

    model_path = ensure_model()
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
    )

    video_dir = Path(args.video_dir)
    action_dirs = sorted([d for d in video_dir.iterdir() if d.is_dir()])
    print(f"Found {len(action_dirs)} action classes: {[d.name for d in action_dirs]}")

    done_clip_ids = set()
    if args.resume and Path(args.out).exists():
        existing = pd.read_parquet(args.out)
        done_clip_ids = set(existing["clip_id"].unique())
        print(f"Resuming — {len(done_clip_ids)} clips already done.")

    all_rows = []
    for action_dir in action_dirs:
        clips = sorted(action_dir.glob("*.avi"))
        for clip_path in tqdm(clips, desc=action_dir.name):
            actor, seq = parse_filename(clip_path)
            if actor is None:
                print(f"  skipping unparseable filename: {clip_path.name}")
                continue
            clip_id = clip_path.stem
            if clip_id in done_clip_ids:
                continue
            rows = extract_clip(clip_path, options, clip_id, actor, action_dir.name, seq)
            all_rows.extend(rows)

    new_df = pd.DataFrame(all_rows)
    if args.resume and Path(args.out).exists() and not new_df.empty:
        existing = pd.read_parquet(args.out)
        new_df = pd.concat([existing, new_df], ignore_index=True)

    new_df.to_parquet(args.out, index=False)
    print(f"Saved {new_df['clip_id'].nunique()} clips, {len(new_df)} keypoint rows -> {args.out}")


if __name__ == "__main__":
    main()
