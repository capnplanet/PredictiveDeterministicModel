from __future__ import annotations

from pathlib import Path

import pytest

from app.db.models import Artifact, Entity, Event, Interaction, ModelRun
from app.db.session import session_scope
from app.services.artifact_ingestion import ingest_artifacts_manifest
from app.services.csv_ingestion import ingest_entities_csv, ingest_events_csv, ingest_interactions_csv
from app.services.feature_extraction import extract_features_for_pending
from app.training.model import EncoderConfig
from app.training.synth_data import generate_synthetic_dataset
from app.training.train import _build_entity_tensors, _build_modality_tensors


@pytest.mark.integration
def test_build_modality_tensors_uses_nonzero_real_signals(tmp_path: Path) -> None:
    synth_dir = tmp_path / "synth"
    generate_synthetic_dataset(
        out_dir=synth_dir,
        n_entities=16,
        n_events=128,
        n_interactions=64,
        n_artifacts=16,
        seed=2024,
    )

    with session_scope() as session:
        session.query(Artifact).delete()
        session.query(ModelRun).delete()
        session.query(Interaction).delete()
        session.query(Event).delete()
        session.query(Entity).delete()

        ingest_entities_csv(session, synth_dir / "entities.csv")
        ingest_events_csv(session, synth_dir / "events.csv")
        ingest_interactions_csv(session, synth_dir / "interactions.csv")
        ingest_artifacts_manifest(session, synth_dir / "artifacts_manifest.csv")
        updated = extract_features_for_pending(session)

    assert updated > 0

    id_to_idx, attr_vec_np, *_ = _build_entity_tensors()
    (
        event_type_ids,
        event_values,
        event_deltas,
        neighbor_attr,
        neighbor_mask,
        artifact_feats,
        coverage,
    ) = _build_modality_tensors(
        id_to_idx=id_to_idx,
        attr_vec_np=attr_vec_np,
        encoder_cfg=EncoderConfig(),
        artifact_feat_dim=32,
    )

    assert coverage["entities_with_events"] > 0
    assert coverage["entities_with_neighbors"] > 0
    assert coverage["entities_with_artifacts"] > 0
    assert coverage["nonzero_event_values"] > 0

    assert int((event_values != 0).sum().item()) > 0
    assert int((event_deltas != 0).sum().item()) > 0
    assert int((neighbor_mask != 0).sum().item()) > 0
    assert int((neighbor_attr != 0).sum().item()) > 0
    assert int((artifact_feats != 0).sum().item()) > 0
    assert int((event_type_ids != 0).sum().item()) > 0
