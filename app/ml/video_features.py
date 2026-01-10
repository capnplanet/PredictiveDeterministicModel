from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def extract_video_features(path: Path) -> Tuple[np.ndarray, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    frame_step = 10
    frame_idx = 0
    per_frame_features: list[np.ndarray] = []
    prev_gray: np.ndarray | None = None
    motion_stats: list[float] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Reuse image features by writing to a temporary in-memory image would be ideal,
            # but to avoid extra IO we compute simplified features here.
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype("float32") / 255.0
            hist_gray, _ = np.histogram(gray, bins=32, range=(0.0, 1.0), density=True)
            feat = hist_gray.astype("float32")
            per_frame_features.append(feat)

            if prev_gray is not None:
                diff = np.abs(gray - prev_gray)
                motion_stats.append(float(diff.mean()))
            prev_gray = gray
        frame_idx += 1

    cap.release()

    if not per_frame_features:
        # Fallback: treat video as a single black frame
        per_frame_features.append(np.zeros(32, dtype="float32"))

    stack = np.stack(per_frame_features, axis=0)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0)
    max_v = stack.max(axis=0)

    motion_arr = np.array(motion_stats or [0.0], dtype="float32")
    motion_mean = np.array([motion_arr.mean()], dtype="float32")
    motion_std = np.array([motion_arr.std()], dtype="float32")

    feat = np.concatenate([mean, std, max_v, motion_mean, motion_std]).astype("float32")
    return feat, int(feat.shape[0])
