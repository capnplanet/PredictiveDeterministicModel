from __future__ import annotations

from pathlib import Path

import numpy as np

from app.ml.audio_features import extract_audio_features
from app.ml.image_features import extract_image_features
from app.ml.video_features import extract_video_features
from app.training.synth_data import _generate_image, _generate_audio, _generate_video_frames, _write_video  # type: ignore[attr-defined]


def test_image_feature_determinism(tmp_path: Path) -> None:
    img = _generate_image(np.array([0.1, -0.2, 0.3]))
    path = tmp_path / "img.png"
    img.save(path)
    f1, d1 = extract_image_features(path)
    f2, d2 = extract_image_features(path)
    assert d1 == d2
    assert np.array_equal(f1, f2)


def test_audio_feature_determinism(tmp_path: Path) -> None:
    audio = _generate_audio(np.array([0.1, -0.2, 0.3]))
    path = tmp_path / "aud.wav"
    import soundfile as sf

    sf.write(str(path), audio, 16000)
    f1, d1 = extract_audio_features(path)
    f2, d2 = extract_audio_features(path)
    assert d1 == d2
    assert np.array_equal(f1, f2)


def test_video_feature_determinism(tmp_path: Path) -> None:
    frames = _generate_video_frames(np.array([0.1, -0.2, 0.3]), frames=4)
    path = tmp_path / "vid.avi"
    _write_video(path, frames)
    f1, d1 = extract_video_features(path)
    f2, d2 = extract_video_features(path)
    assert d1 == d2
    assert np.array_equal(f1, f2)
