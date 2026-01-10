from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from skimage import color, feature


def _load_image(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img = img.resize((128, 128), resample=Image.BILINEAR)
    return img


def extract_image_features(path: Path) -> Tuple[np.ndarray, int]:
    img = _load_image(path)
    arr = np.asarray(img).astype("float32") / 255.0

    gray = color.rgb2gray(arr)

    # Grayscale histogram
    hist_gray, _ = np.histogram(gray, bins=32, range=(0.0, 1.0), density=True)

    # RGB histogram
    hist_r, _ = np.histogram(arr[:, :, 0], bins=32, range=(0.0, 1.0), density=True)
    hist_g, _ = np.histogram(arr[:, :, 1], bins=32, range=(0.0, 1.0), density=True)
    hist_b, _ = np.histogram(arr[:, :, 2], bins=32, range=(0.0, 1.0), density=True)

    # HOG features
    hog_vec, _ = feature.hog(
        gray,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        feature_vector=True,
    )

    # Edge density via Canny
    edges = feature.canny(gray, sigma=1.0)
    edge_density = np.array([edges.mean()], dtype="float32")

    feat = np.concatenate(
        [hist_gray, hist_r, hist_g, hist_b, hog_vec.astype("float32"), edge_density]
    )
    return feat.astype("float32"), int(feat.shape[0])
