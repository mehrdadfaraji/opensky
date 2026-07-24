"""
Data-understanding tool: overlay the extracted keypoints back onto the
original video, frame by frame, so you can *see* the pixels-to-numbers
transformation instead of just reading two disconnected tables.

Usage:
    python visualize_clip.py --clip-id p10_smash_s1 \
        --video data/thetis/VIDEO_RGB/smash/p10_smash_s1.avi \
        --keypoints keypoints.parquet \
        --out p10_smash_s1_annotated.mp4

Output: an .mp4, same length/fps as the original, with the 33 landmarks
drawn as dots and a simplified skeleton drawn as lines on top of each frame.
Landmarks below --min-visibility are drawn in red instead of green, so you
can see visually where the visibility discussion actually shows up.
"""

import argparse

import cv2
import pandas as pd

# A simplified skeleton — just the segments relevant to a tennis swing.
# (landmark indices per MediaPipe's standard 33-point pose model)
CONNECTIONS = [
    (11, 12),  # shoulders
    (11, 13), (13, 15),  # left shoulder -> elbow -> wrist
    (12, 14), (14, 16),  # right shoulder -> elbow -> wrist
    (11, 23), (12, 24),  # shoulders -> hips
    (23, 24),            # hips
    (23, 25), (25, 27),  # left hip -> knee -> ankle
    (24, 26), (26, 28),  # right hip -> knee -> ankle
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--keypoints", default="keypoints.parquet")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-visibility", type=float, default=0.5)
    args = ap.parse_args()
    out_path = args.out or f"{args.clip_id}_annotated.mp4"

    df = pd.read_parquet(args.keypoints)
    clip_df = df[df["clip_id"] == args.clip_id]
    if clip_df.empty:
        raise SystemExit(f"No rows found for clip_id={args.clip_id} in {args.keypoints}")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = 0
    frames_with_detection = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_kp = clip_df[clip_df["frame_idx"] == frame_idx]
        if not frame_kp.empty:
            frames_with_detection += 1
            points = {}
            for _, row in frame_kp.iterrows():
                px, py = int(row["x"] * w), int(row["y"] * h)
                points[int(row["landmark_id"])] = (px, py, row["visibility"])
                color = (0, 255, 0) if row["visibility"] >= args.min_visibility else (0, 0, 255)
                cv2.circle(frame, (px, py), 4, color, -1)

            for a, b in CONNECTIONS:
                if a in points and b in points:
                    cv2.line(frame, points[a][:2], points[b][:2], (255, 255, 255), 1)
        else:
            cv2.putText(frame, "NO DETECTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

        cv2.putText(frame, f"frame {frame_idx}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"{frames_with_detection}/{frame_idx} frames had a detection.")
    print(f"Saved annotated video -> {out_path}")


if __name__ == "__main__":
    main()
