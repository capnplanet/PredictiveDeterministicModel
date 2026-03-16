from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from app.db.models import Entity
from app.db.session import session_scope
from app.services.artifact_ingestion import ingest_artifact_file
from app.services.csv_ingestion import ingest_entities_csv
from app.services.feature_extraction import extract_features_for_artifact


@pytest.mark.integration
def test_entities_ingestion_checkpoint_resume(tmp_path: Path) -> None:
    csv_path = tmp_path / "entities.csv"
    checkpoint = tmp_path / "checkpoints" / "entities_checkpoint.json"
    suffix = uuid.uuid4().hex[:8]

    rows = [
        (f"E_CP_{suffix}_001", {"x": 0.1, "y": 0.2, "z": 0.3}),
        (f"E_CP_{suffix}_002", {"x": 0.2, "y": 0.3, "z": 0.4}),
        (f"E_CP_{suffix}_003", {"x": 0.3, "y": 0.4, "z": 0.5}),
    ]
    csv_lines = ["entity_id,attributes,created_at"]
    for idx, (entity_id, attrs) in enumerate(rows):
        attrs_field = '"' + json.dumps(attrs).replace('"', '""') + '"'
        csv_lines.append(f"{entity_id},{attrs_field},2025-01-01T00:00:0{idx}")
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    with session_scope() as session:
        report_first = ingest_entities_csv(
            session,
            csv_path,
            chunk_size=1,
            checkpoint_path=checkpoint,
            resume_from_checkpoint=True,
        )
        assert report_first.success_rows == 3

    checkpoint.write_text(json.dumps({"last_processed_row": 2}), encoding="utf-8")

    with session_scope() as session:
        report_second = ingest_entities_csv(
            session,
            csv_path,
            chunk_size=1,
            checkpoint_path=checkpoint,
            resume_from_checkpoint=True,
        )
        assert report_second.success_rows == 1

    with session_scope() as session:
        all_rows = session.query(Entity).filter(Entity.entity_id.like(f"E_CP_{suffix}_%")).all()
        assert len(all_rows) == 3


@pytest.mark.integration
def test_feature_extraction_invalidates_stale_feature_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from app.services import feature_extraction as feature_module

    image_path = tmp_path / "sample.png"
    from PIL import Image

    Image.new("RGB", (2, 2), color=(120, 30, 200)).save(image_path)

    with session_scope() as session:
        session.merge(
            Entity(
                entity_id="E_CP_IMG",
                attributes={"x": 0.0, "y": 0.0, "z": 0.0},
            )
        )
        artifact = ingest_artifact_file(session, image_path, "image", entity_id="E_CP_IMG")
        artifact.feature_status = "done"  # type: ignore[assignment]
        artifact.feature_version_hash = "stale_version"
        session.flush()

        monkeypatch.setattr(feature_module, "compute_feature_version_hash", lambda: "fresh_version")
        monkeypatch.setattr(
            feature_module,
            "extract_image_features",
            lambda _path: (np.arange(8, dtype="float32"), 8),
        )

        extract_features_for_artifact(session, artifact)
        assert str(artifact.feature_status) == "done"
        assert artifact.feature_version_hash == "fresh_version"
        assert artifact.feature_dim == 8
