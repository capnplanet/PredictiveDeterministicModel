from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter
from PIL import Image

from app.api.schemas import (
    ArtifactIngestionReportModel,
    DemoPreloadRequest,
    DemoPreloadResponse,
    IngestionReportModel,
)
from app.core.config import get_settings
from app.db.models import Artifact, Entity, Event, Interaction, ModelRun
from app.db.session import session_scope
from app.services.artifact_ingestion import ingest_artifact_file, ingest_artifacts_manifest
from app.services.csv_ingestion import ingest_entities_csv, ingest_events_csv, ingest_interactions_csv
from app.services.feature_extraction import extract_features_for_pending
from app.training.synth_data import generate_synthetic_dataset
from app.training.train import run_training

router = APIRouter(prefix="/demo", tags=["demo"])


def _profile_sizes(profile: str) -> tuple[int, int, int, int]:
    if profile == "medium":
        return 48, 360, 160, 36
    return 18, 120, 48, 12


def _reset_existing_state() -> None:
    settings = get_settings()
    with session_scope() as session:
        session.query(Artifact).delete()
        session.query(ModelRun).delete()
        session.query(Interaction).delete()
        session.query(Event).delete()
        session.query(Entity).delete()

    artifacts_store = Path(settings.data_root) / "artifacts_store"
    feature_cache = Path(settings.data_root) / "feature_cache"
    uploads = Path(settings.data_root) / "uploads"
    for root in (artifacts_store, feature_cache, uploads):
        root.mkdir(parents=True, exist_ok=True)
        for child in root.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                import shutil

                shutil.rmtree(child, ignore_errors=True)


@router.post("/preload", response_model=DemoPreloadResponse)
async def preload_demo_data(request: DemoPreloadRequest) -> DemoPreloadResponse:
    settings = get_settings()
    n_entities, n_events, n_interactions, n_artifacts = _profile_sizes(request.profile)

    if request.reset_existing:
        _reset_existing_state()

    out_dir = Path(settings.data_root) / "demo_preload" / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    generate_synthetic_dataset(
        out_dir=out_dir,
        n_entities=n_entities,
        n_events=n_events,
        n_interactions=n_interactions,
        n_artifacts=n_artifacts,
        seed=42,
    )

    with session_scope() as session:
        entities_report = ingest_entities_csv(session, out_dir / "entities.csv")
        events_report = ingest_events_csv(session, out_dir / "events.csv")
        interactions_report = ingest_interactions_csv(session, out_dir / "interactions.csv")
        artifacts_report = ingest_artifacts_manifest(session, out_dir / "artifacts_manifest.csv")

        # Ensure single-artifact ingestion path is also represented.
        single_path = out_dir / "single_demo_upload.png"
        Image.new("RGB", (32, 32), color=(80, 120, 180)).save(single_path)
        single_artifact = ingest_artifact_file(
            session=session,
            source_path=single_path,
            artifact_type="image",
            entity_id="E00000",
            timestamp=datetime.utcnow(),
            metadata={"source": "demo_preload_single"},
        )

        updated = 0
        if request.extract_features:
            updated = extract_features_for_pending(session)

    training_summary = None
    if request.train_model:
        import json

        quick_train_config_path = Path(settings.data_root) / "demo_preload_train_config.json"
        quick_train_config_path.parent.mkdir(parents=True, exist_ok=True)
        quick_train_config_path.write_text(
            json.dumps(
                {
                    "epochs": 1,
                    "batch_size": 16,
                    "lr": 0.001,
                    "seed": 1234,
                    "val_fraction": 0.2,
                    "test_fraction": 0.2,
                    "split_strategy": "random",
                    "corpus_name": "demo_preload",
                    "threshold_policy_version": "v1",
                    "enforce_thresholds": False,
                }
            )
        )
        run_id, metrics = run_training(config_path=quick_train_config_path)
        training_summary = {"run_id": run_id, "metrics": metrics}

    return DemoPreloadResponse(
        profile=request.profile,
        output_dir=str(out_dir),
        entities=IngestionReportModel(**entities_report.__dict__),
        events=IngestionReportModel(**events_report.__dict__),
        interactions=IngestionReportModel(**interactions_report.__dict__),
        artifacts_manifest=ArtifactIngestionReportModel(**artifacts_report.__dict__),
        single_artifact={
            "artifact_id": str(single_artifact.artifact_id),
            "sha256": str(single_artifact.sha256),
            "artifact_type": str(single_artifact.artifact_type),
        },
        features={"updated_artifacts": updated},
        training=training_summary,
        sample_entity_ids=[f"E{i:05d}" for i in range(min(5, n_entities))],
    )
