from __future__ import annotations

from pathlib import Path
from typing import Tuple

import librosa
import numpy as np


def extract_audio_features(path: Path) -> Tuple[np.ndarray, int]:
    # Deterministic loading: fixed sr and duration
    y, sr = librosa.load(path, sr=16000, mono=True)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    # Log-mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_mean = mel_db.mean(axis=1)
    mel_std = mel_db.std(axis=1)

    # Simple statistics
    rms = librosa.feature.rms(y=y).mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y=y).mean(axis=1)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean(axis=1)

    feat = np.concatenate([
        mfcc_mean,
        mfcc_std,
        mel_mean,
        mel_std,
        rms,
        zcr,
        centroid,
    ]).astype("float32")
    return feat, int(feat.shape[0])
