from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Artifact, ArtifactType


class ArtifactIngestionError(Exception):
    pass


@dataclass
class ArtifactIngestionReport:
    total_rows: int
    success_rows: int
    failed_rows: int
    errors: List[str]


_MAX_ROWS = 100_000
_MAX_FILE_BYTES = 200 * 1024 * 1024


def _artifacts_root() -> Path:
    settings = get_settings()
    root = Path(settings.data_root) / "artifacts_store"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _validate_artifact_file(path: Path) -> None:
    if not path.exists():
        raise ArtifactIngestionError(f"Artifact file not found: {path}")
    size = path.stat().st_size
    if size > _MAX_FILE_BYTES:
        raise ArtifactIngestionError(f"Artifact file too large: {size} bytes > max {_MAX_FILE_BYTES}")


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest_artifact_file(
    session: Session,
    source_path: Path,
    artifact_type: str,
    entity_id: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    metadata: Optional[dict] = None,
) -> Artifact:
    _validate_artifact_file(source_path)
    root = _artifacts_root()
    sha256 = _compute_sha256(source_path)
    dest_path = root / sha256
    if not dest_path.exists():
        dest_path.write_bytes(source_path.read_bytes())

    atype = ArtifactType.python_type(artifact_type) if hasattr(ArtifactType, "python_type") else artifact_type  # type: ignore[assignment]

    artifact = Artifact(
        entity_id=entity_id,
        timestamp=timestamp,
        artifact_type=atype,  # type: ignore[arg-type]
        file_path=str(dest_path),
        sha256=sha256,
        metadata=metadata or {},
    )
    session.add(artifact)
    session.flush()
    return artifact


def ingest_artifacts_manifest(session: Session, manifest_path: Path) -> ArtifactIngestionReport:
    if not manifest_path.exists():
        raise ArtifactIngestionError(f"Manifest not found: {manifest_path}")

    total = success = failed = 0
    errors: List[str] = []

    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if total > _MAX_ROWS:
                raise ArtifactIngestionError(f"Row limit exceeded: {_MAX_ROWS}")
            try:
                atype = row["artifact_type"].strip()
                if atype not in {"image", "video", "audio"}:
                    raise ValueError(f"invalid artifact_type {atype}")
                rel_path = row["path"].strip()
                src = Path(rel_path).expanduser().resolve()
                ts_raw = row.get("timestamp", "").strip()
                ts = datetime.fromisoformat(ts_raw) if ts_raw else None
                entity_id = row.get("entity_id", "").strip() or None
                metadata_raw = row.get("metadata", "").strip()
                metadata = json.loads(metadata_raw) if metadata_raw else {}
                ingest_artifact_file(
                    session=session,
                    source_path=src,
                    artifact_type=atype,
                    entity_id=entity_id,
                    timestamp=ts,
                    metadata=metadata,
                )
                success += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                errors.append(f"Row {total}: {exc}")

    return ArtifactIngestionReport(total_rows=total, success_rows=success, failed_rows=failed, errors=errors)
