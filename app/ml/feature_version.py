from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict

import librosa
import numpy as np
import PIL
import skimage
import torch


@dataclass(frozen=True)
class FeatureVersionInfo:
    code_hash: str
    config: Dict[str, float]
    library_versions: Dict[str, str]

    def to_hash(self) -> str:
        payload = {
            "code_hash": self.code_hash,
            "config": self.config,
            "library_versions": self.library_versions,
        }
        data = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(data).hexdigest()


def compute_feature_version_hash() -> str:
    # In a real system, code_hash would be derived from the actual source files.
    # Here we fix a constant string but include library versions and config so that
    # any meaningful change will alter the version hash deterministically.
    code_hash = "v1_multimodal_feature_pipeline"
    config = {
        "image_hist_bins": 32.0,
        "hog_pixels_per_cell": 8.0,
        "hog_cells_per_block": 2.0,
        "audio_mfcc": 20.0,
        "audio_mels": 40.0,
        "video_frame_step": 10.0,
    }
    versions = {
        "numpy": np.__version__,
        "torch": torch.__version__,
        "librosa": librosa.__version__,
        "pillow": PIL.__version__,
        "skimage": skimage.__version__,
    }
    info = FeatureVersionInfo(code_hash=code_hash, config=config, library_versions=versions)
    return info.to_hash()
