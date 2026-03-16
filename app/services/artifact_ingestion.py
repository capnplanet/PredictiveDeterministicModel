from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import List, Optional

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.performance import emit_performance_event
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
_DEFAULT_CHUNK_SIZE = 250


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


def _load_checkpoint(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return max(0, int(payload.get("last_processed_row", 0)))
    except Exception as exc:  # noqa: BLE001
        raise ArtifactIngestionError(f"Invalid checkpoint file: {path}: {exc}") from exc


def _write_checkpoint(path: Path, last_processed_row: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"last_processed_row": int(max(0, last_processed_row))}, sort_keys=True),
        encoding="utf-8",
    )


def ingest_artifact_file(
    session: Session,
    source_path: Path,
    artifact_type: str,
    entity_id: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    metadata: Optional[dict] = None,
) -> Artifact:
    started = perf_counter()
    _validate_artifact_file(source_path)
    source_size = source_path.stat().st_size
    root = _artifacts_root()
    sha256 = _compute_sha256(source_path)
    dest_path = root / sha256
    existed_before = dest_path.exists()
    if not dest_path.exists():
        dest_path.write_bytes(source_path.read_bytes())

    atype = (
        ArtifactType.python_type(artifact_type)
        if hasattr(ArtifactType, "python_type")
        else artifact_type
    )  # type: ignore[assignment]

    artifact = Artifact(
        entity_id=entity_id,
        timestamp=timestamp,
        artifact_type=atype,  # type: ignore[arg-type]
        file_path=str(dest_path),
        sha256=sha256,
        meta=metadata or {},
    )
    session.add(artifact)
    session.flush()
    duration_ms = (perf_counter() - started) * 1000.0
    throughput = source_size / max(duration_ms / 1000.0, 1e-9)
    emit_performance_event(
        "ingestion.artifact_file",
        duration_ms=duration_ms,
        artifact_type=artifact_type,
        source_size_bytes=source_size,
        throughput_bytes_per_sec=round(throughput, 3),
        cache_hit=existed_before,
    )
    return artifact


def ingest_artifacts_manifest(
    session: Session,
    manifest_path: Path,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    checkpoint_path: Path | None = None,
    resume_from_checkpoint: bool = True,
) -> ArtifactIngestionReport:
    started = perf_counter()
    if not manifest_path.exists():
        raise ArtifactIngestionError(f"Manifest not found: {manifest_path}")
    if chunk_size <= 0:
        raise ArtifactIngestionError("chunk_size must be > 0")

    total = success = failed = 0
    errors: List[str] = []
    checkpoint_row = (
        _load_checkpoint(checkpoint_path) if checkpoint_path is not None and resume_from_checkpoint else 0
    )
    last_processed_row = checkpoint_row
    flush_counter = 0

    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_number, row in enumerate(reader, start=1):
            if row_number <= checkpoint_row:
                continue

            total += 1
            if row_number > _MAX_ROWS:
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
                flush_counter += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                errors.append(f"Row {total}: {exc}")
            finally:
                last_processed_row = row_number

            if flush_counter >= int(chunk_size):
                session.flush()
                flush_counter = 0
                if checkpoint_path is not None:
                    _write_checkpoint(checkpoint_path, last_processed_row)

    if success:
        session.flush()
    if checkpoint_path is not None:
        _write_checkpoint(checkpoint_path, last_processed_row)

    duration_ms = (perf_counter() - started) * 1000.0
    throughput = (success / max(duration_ms / 1000.0, 1e-9)) if success else 0.0
    emit_performance_event(
        "ingestion.artifacts_manifest",
        duration_ms=duration_ms,
        manifest_path=str(manifest_path),
        total_rows=total,
        success_rows=success,
        failed_rows=failed,
        chunk_size=int(chunk_size),
        resumed_from_row=checkpoint_row,
        throughput_rows_per_sec=round(throughput, 3),
    )
    return ArtifactIngestionReport(total_rows=total, success_rows=success, failed_rows=failed, errors=errors)
