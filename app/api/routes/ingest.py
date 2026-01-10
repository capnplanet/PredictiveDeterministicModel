from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Form

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


@router.post("/entities", response_model=IngestionReportModel)
async def ingest_entities_csv_api(file: UploadFile = File(...)) -> IngestionReportModel:
    path = _save_temp_file(file, "entities")
    with session_scope() as session:
        report = ingest_entities_csv(session, path)
    return IngestionReportModel(**report.__dict__)


@router.post("/events", response_model=IngestionReportModel)
async def ingest_events_csv_api(file: UploadFile = File(...)) -> IngestionReportModel:
    path = _save_temp_file(file, "events")
    with session_scope() as session:
        report = ingest_events_csv(session, path)
    return IngestionReportModel(**report.__dict__)


@router.post("/interactions", response_model=IngestionReportModel)
async def ingest_interactions_csv_api(file: UploadFile = File(...)) -> IngestionReportModel:
    path = _save_temp_file(file, "interactions")
    with session_scope() as session:
        report = ingest_interactions_csv(session, path)
    return IngestionReportModel(**report.__dict__)


@router.post("/artifacts", response_model=ArtifactIngestionReportModel)
async def ingest_artifacts_manifest_api(file: UploadFile = File(...)) -> ArtifactIngestionReportModel:
    path = _save_temp_file(file, "artifacts_manifest")
    with session_scope() as session:
        report = ingest_artifacts_manifest(session, path)
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
    from datetime import datetime
    import json

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
