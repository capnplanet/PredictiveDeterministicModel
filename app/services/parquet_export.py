from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd

from app.db.session import get_engine


def export_parquet(out_dir: Path) -> dict:
    """Export core tables to Parquet files in out_dir.

    - entities.parquet
    - events.parquet
    - interactions.parquet
    - artifacts.parquet (metadata and feature references only)
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    engine = get_engine()

    for table_name in ["entities", "events", "interactions", "artifacts"]:
        df = pd.read_sql_table(table_name, con=engine)

        # Convert UUID/object columns to strings so pyarrow can serialize them
        for col in df.columns:
            if df[col].dtype == object:
                sample = next((v for v in df[col] if v is not None), None)
                if isinstance(sample, uuid.UUID):
                    df[col] = df[col].astype(str)

        df.to_parquet(out_dir / f"{table_name}.parquet")

    return {
        "out_dir": str(out_dir),
        "files": [
            str(out_dir / "entities.parquet"),
            str(out_dir / "events.parquet"),
            str(out_dir / "interactions.parquet"),
            str(out_dir / "artifacts.parquet"),
        ],
    }
