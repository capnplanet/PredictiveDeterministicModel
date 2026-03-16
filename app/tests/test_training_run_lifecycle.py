from __future__ import annotations

import json
from pathlib import Path

import pytest

import app.training.train as train_module
from app.db.models import Artifact, Entity, Event, Interaction, ModelRun
from app.db.session import session_scope
from app.services.csv_ingestion import ingest_entities_csv, ingest_events_csv, ingest_interactions_csv
from app.training.train import run_training


def _seed_minimal_training_data(tmp_path: Path) -> None:
    entities_csv = tmp_path / "entities.csv"
    events_csv = tmp_path / "events.csv"
    interactions_csv = tmp_path / "interactions.csv"

    entities_csv.write_text(
        "entity_id,attributes,created_at\n"
        'LIFE_A,"{""x"":0.1,""y"":0.2,""z"":0.3,""target_regression"":0.8,""target_binary"":1,""target_ranking"":0.7}",2025-01-01T00:00:00\n'
        'LIFE_B,"{""x"":0.3,""y"":0.1,""z"":0.4,""target_regression"":0.2,""target_binary"":0,""target_ranking"":0.1}",2025-01-01T00:01:00\n',
        encoding="utf-8",
    )
    events_csv.write_text(
        "timestamp,entity_id,event_type,event_value,event_metadata\n"
        '2025-01-01T00:10:00,LIFE_A,purchase,10.0,"{""src"":""lifecycle""}"\n'
        '2025-01-01T00:11:00,LIFE_B,view,3.5,"{""src"":""lifecycle""}"\n',
        encoding="utf-8",
    )
    interactions_csv.write_text(
        "timestamp,src_entity_id,dst_entity_id,interaction_type,interaction_value,metadata\n"
        '2025-01-01T00:12:00,LIFE_A,LIFE_B,link,1.0,"{""src"":""lifecycle""}"\n',
        encoding="utf-8",
    )

    with session_scope() as session:
        session.query(ModelRun).delete()
        session.query(Artifact).delete()
        session.query(Interaction).delete()
        session.query(Event).delete()
        session.query(Entity).delete()

        ingest_entities_csv(session, entities_csv)
        ingest_events_csv(session, events_csv)
        ingest_interactions_csv(session, interactions_csv)


@pytest.mark.integration
def test_run_training_marks_run_success_after_pending(tmp_path: Path) -> None:
    _seed_minimal_training_data(tmp_path)

    cfg_path = tmp_path / "train_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "epochs": 1,
                "batch_size": 2,
                "seed": 2026,
                "val_fraction": 0.2,
                "test_fraction": 0.2,
            }
        ),
        encoding="utf-8",
    )

    run_id, _ = run_training(config_path=cfg_path)

    with session_scope() as session:
        run = session.get(ModelRun, run_id)
        assert run is not None
        assert str(run.status) == "success"
        assert run.model_sha256 != "pending"


@pytest.mark.integration
def test_run_training_marks_failed_when_artifact_persistence_breaks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_minimal_training_data(tmp_path)

    cfg_path = tmp_path / "train_config_fail.json"
    cfg_path.write_text(
        json.dumps(
            {
                "epochs": 1,
                "batch_size": 2,
                "seed": 3031,
                "val_fraction": 0.2,
                "test_fraction": 0.2,
            }
        ),
        encoding="utf-8",
    )

    captured_run_id: dict[str, str] = {}
    original_build_run_id = train_module._build_run_id

    def _capture_run_id(*args, **kwargs):  # type: ignore[no-untyped-def]
        rid = original_build_run_id(*args, **kwargs)
        captured_run_id["value"] = rid
        return rid

    def _raise_on_torch_save(*_args, **_kwargs) -> None:  # type: ignore[no-untyped-def]
        raise RuntimeError("simulated artifact persistence failure")

    monkeypatch.setattr(train_module, "_build_run_id", _capture_run_id)
    monkeypatch.setattr(train_module.torch, "save", _raise_on_torch_save)

    with pytest.raises(RuntimeError, match="simulated artifact persistence failure"):
        run_training(config_path=cfg_path)

    assert "value" in captured_run_id

    with session_scope() as session:
        run = session.get(ModelRun, captured_run_id["value"])
        assert run is not None
        assert str(run.status) == "failed"
        assert float(run.metrics.get("persistence_failed", 0.0)) == 1.0
        assert "persistence_error" in run.data_manifest
