from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile

from app.api.schemas import ArtifactIngestionReportModel, IngestionReportModel
from app.core.config import get_settings
from app.db.session import session_scope
from app.services.artifact_ingestion import ingest_artifact_file, ingest_artifacts_manifest
from app.services.csv_ingestion import (
    ingest_entities_csv,
    ingest_events_csv,
    ingest_interactions_csv,
)

router = APIRouter(prefix="/ingest", tags=["ingest"])


def _save_temp_file(upload: UploadFile, subdir: str) -> Path:
    settings = get_settings()
    root = Path(settings.data_root) / "uploads" / subdir
    root.mkdir(parents=True, exist_ok=True)
    dest = root / upload.filename
    with dest.open("wb") as out_f:
        shutil.copyfileobj(upload.file, out_f)
    return dest


def _checkpoint_path(kind: str, checkpoint_key: Optional[str]) -> Optional[Path]:
    key = (checkpoint_key or "").strip()
    if not key:
        return None
    settings = get_settings()
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    path = Path(settings.data_root) / "checkpoints" / kind / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@router.post("/entities", response_model=IngestionReportModel)
async def ingest_entities_csv_api(
    file: UploadFile = File(...),
    chunk_size: int = Form(1000),
    checkpoint_key: Optional[str] = Form(None),
    resume_from_checkpoint: bool = Form(True),
) -> IngestionReportModel:
    path = _save_temp_file(file, "entities")
    checkpoint_path = _checkpoint_path("entities", checkpoint_key)
    with session_scope() as session:
        report = ingest_entities_csv(
            session,
            path,
            chunk_size=chunk_size,
            checkpoint_path=checkpoint_path,
            resume_from_checkpoint=resume_from_checkpoint,
        )
    return IngestionReportModel(**report.__dict__)


@router.post("/events", response_model=IngestionReportModel)
async def ingest_events_csv_api(
    file: UploadFile = File(...),
    chunk_size: int = Form(1000),
    checkpoint_key: Optional[str] = Form(None),
    resume_from_checkpoint: bool = Form(True),
) -> IngestionReportModel:
    path = _save_temp_file(file, "events")
    checkpoint_path = _checkpoint_path("events", checkpoint_key)
    with session_scope() as session:
        report = ingest_events_csv(
            session,
            path,
            chunk_size=chunk_size,
            checkpoint_path=checkpoint_path,
            resume_from_checkpoint=resume_from_checkpoint,
        )
    return IngestionReportModel(**report.__dict__)


@router.post("/interactions", response_model=IngestionReportModel)
async def ingest_interactions_csv_api(
    file: UploadFile = File(...),
    chunk_size: int = Form(1000),
    checkpoint_key: Optional[str] = Form(None),
    resume_from_checkpoint: bool = Form(True),
) -> IngestionReportModel:
    path = _save_temp_file(file, "interactions")
    checkpoint_path = _checkpoint_path("interactions", checkpoint_key)
    with session_scope() as session:
        report = ingest_interactions_csv(
            session,
            path,
            chunk_size=chunk_size,
            checkpoint_path=checkpoint_path,
            resume_from_checkpoint=resume_from_checkpoint,
        )
    return IngestionReportModel(**report.__dict__)


@router.post("/artifacts", response_model=ArtifactIngestionReportModel)
async def ingest_artifacts_manifest_api(
    file: UploadFile = File(...),
    chunk_size: int = Form(250),
    checkpoint_key: Optional[str] = Form(None),
    resume_from_checkpoint: bool = Form(True),
) -> ArtifactIngestionReportModel:
    path = _save_temp_file(file, "artifacts_manifest")
    checkpoint_path = _checkpoint_path("artifacts", checkpoint_key)
    with session_scope() as session:
        report = ingest_artifacts_manifest(
            session,
            path,
            chunk_size=chunk_size,
            checkpoint_path=checkpoint_path,
            resume_from_checkpoint=resume_from_checkpoint,
        )
    return ArtifactIngestionReportModel(**report.__dict__)


@router.post("/artifact", response_model=dict)
async def ingest_single_artifact(
    file: UploadFile = File(...),
    artifact_type: str = Form(...),
    entity_id: Optional[str] = Form(None),
    timestamp: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
) -> dict:
    temp_path = _save_temp_file(file, "artifact_single")
    import json
    from datetime import datetime

    ts_val = datetime.fromisoformat(timestamp) if timestamp else None
    meta_val = json.loads(metadata) if metadata else None

    with session_scope() as session:
        artifact = ingest_artifact_file(
            session=session,
            source_path=temp_path,
            artifact_type=artifact_type,
            entity_id=entity_id,
            timestamp=ts_val,
            metadata=meta_val,
        )
        session.flush()
        return {
            "artifact_id": str(artifact.artifact_id),
            "sha256": artifact.sha256,
            "artifact_type": str(artifact.artifact_type),
        }
