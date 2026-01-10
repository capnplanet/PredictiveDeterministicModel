from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Artifact, FeatureStatus
from app.ml.audio_features import extract_audio_features
from app.ml.feature_version import compute_feature_version_hash
from app.ml.image_features import extract_image_features
from app.ml.video_features import extract_video_features


class FeatureExtractionError(Exception):
    pass


def _cache_root() -> Path:
    settings = get_settings()
    root = Path(settings.data_root) / "feature_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _feature_cache_path(sha256: str) -> Path:
    return _cache_root() / f"{sha256}.npy"


def extract_features_for_artifact(session: Session, artifact: Artifact) -> None:
    cache_path = _feature_cache_path(artifact.sha256)
    if cache_path.exists() and artifact.feature_status == "done":  # type: ignore[comparison-overlap]
        return

    path = Path(artifact.file_path)
    if artifact.artifact_type == "image":  # type: ignore[comparison-overlap]
        vec, dim = extract_image_features(path)
    elif artifact.artifact_type == "audio":  # type: ignore[comparison-overlap]
        vec, dim = extract_audio_features(path)
    elif artifact.artifact_type == "video":  # type: ignore[comparison-overlap]
        vec, dim = extract_video_features(path)
    else:
        raise FeatureExtractionError(f"Unsupported artifact_type {artifact.artifact_type}")

    cache_path.write_bytes(np.asarray(vec, dtype="float32").tobytes())
    version_hash = compute_feature_version_hash()
    artifact.feature_dim = dim
    artifact.feature_version_hash = version_hash
    artifact.feature_status = "done"  # type: ignore[assignment]
    session.add(artifact)


def extract_features_for_pending(session: Session) -> int:
    pending: List[Artifact] = (
        session.query(Artifact).filter(Artifact.feature_status == "pending").all()  # type: ignore[comparison-overlap]
    )
    count = 0
    for art in pending:
        extract_features_for_artifact(session, art)
        count += 1
    return count
