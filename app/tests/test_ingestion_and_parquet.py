from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.db.models import Entity
from app.db.session import session_scope
from app.services.csv_ingestion import ingest_entities_csv
from app.services.parquet_export import export_parquet


@pytest.mark.integration
def test_ingest_and_parquet_export(tmp_path: Path) -> None:
    csv_path = tmp_path / "entities.csv"
    attrs = {
        "x": 0.1,
        "y": 0.2,
        "z": 0.3,
        "target_regression": 0.5,
        "target_binary": 1,
        "target_ranking": 0.5,
    }
    attrs_json = json.dumps(attrs)
    attrs_field = '"' + attrs_json.replace('"', '""') + '"'
    csv_path.write_text(
        "entity_id,attributes,created_at\n"
        f"E00001,{attrs_field},2025-01-01T00:00:00\n"
    )

    with session_scope() as session:
        report = ingest_entities_csv(session, csv_path)
        assert report.success_rows == 1
        ent = session.query(Entity).filter_by(entity_id="E00001").one()
        assert ent.attributes["x"] == 0.1

    out_dir = tmp_path / "parquet"
    export_parquet(out_dir)
    for name in ["entities.parquet", "events.parquet", "interactions.parquet", "artifacts.parquet"]:
        assert (out_dir / name).exists()

    # Sanity check that the exported entities table contains our row
    import pandas as pd

    df = pd.read_parquet(out_dir / "entities.parquet")
    assert "E00001" in set(df["entity_id"])
