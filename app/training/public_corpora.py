from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def _iso_from_unix(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=UTC).replace(tzinfo=None).isoformat()


def _write_empty_artifacts_manifest(out_dir: Path) -> None:
    manifest = out_dir / "artifacts_manifest.csv"
    manifest.write_text("artifact_type,path,entity_id,timestamp,metadata\n", encoding="utf-8")


def prepare_movielens_100k(input_path: Path, out_dir: Path) -> dict[str, int]:
    """Prepare MovieLens 100k interactions into ingestion-ready CSV files.

    Expected input format: `u.data` from ml-100k with tab-separated columns:
    user_id, item_id, rating, timestamp.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(
        input_path,
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
    )

    # Aggregate deterministic user/item targets.
    user_mean = df.groupby("user_id")["rating"].mean().to_dict()
    item_mean = df.groupby("item_id")["rating"].mean().to_dict()

    entities_path = out_dir / "entities.csv"
    with entities_path.open("w", encoding="utf-8") as f:
        f.write("entity_id,attributes,created_at\n")

        for user_id in sorted(user_mean):
            mean_rating = float(user_mean[user_id])
            attrs = {
                "x": mean_rating / 5.0,
                "y": float(df.loc[df["user_id"] == user_id, "rating"].std(ddof=0) or 0.0) / 5.0,
                "z": float(df.loc[df["user_id"] == user_id, "rating"].count()) / max(1.0, float(len(df))),
                "target_regression": mean_rating,
                "target_binary": 1 if mean_rating >= 3.5 else 0,
                "target_ranking": mean_rating,
            }
            attrs_json = json.dumps(attrs, separators=(",", ":")).replace('"', '""')
            created_at = _iso_from_unix(int(df.loc[df["user_id"] == user_id, "timestamp"].min()))
            f.write(f'U{int(user_id):05d},"{attrs_json}",{created_at}\n')

        for item_id in sorted(item_mean):
            mean_rating = float(item_mean[item_id])
            attrs = {
                "x": mean_rating / 5.0,
                "y": float(df.loc[df["item_id"] == item_id, "rating"].std(ddof=0) or 0.0) / 5.0,
                "z": float(df.loc[df["item_id"] == item_id, "rating"].count()) / max(1.0, float(len(df))),
                "target_regression": mean_rating,
                "target_binary": 1 if mean_rating >= 3.5 else 0,
                "target_ranking": mean_rating,
            }
            attrs_json = json.dumps(attrs, separators=(",", ":")).replace('"', '""')
            created_at = _iso_from_unix(int(df.loc[df["item_id"] == item_id, "timestamp"].min()))
            f.write(f'I{int(item_id):05d},"{attrs_json}",{created_at}\n')

    events_path = out_dir / "events.csv"
    with events_path.open("w", encoding="utf-8") as f:
        f.write("timestamp,entity_id,event_type,event_value,event_metadata\n")
        for row in df.itertuples(index=False):
            ts = _iso_from_unix(int(row.timestamp))
            meta = json.dumps({"item_id": int(row.item_id)}, separators=(",", ":")).replace('"', '""')
            f.write(f'{ts},U{int(row.user_id):05d},rating,{float(row.rating)},"{meta}"\n')

    interactions_path = out_dir / "interactions.csv"
    with interactions_path.open("w", encoding="utf-8") as f:
        f.write("timestamp,src_entity_id,dst_entity_id,interaction_type,interaction_value,metadata\n")
        for row in df.itertuples(index=False):
            ts = _iso_from_unix(int(row.timestamp))
            meta = json.dumps({"source": "movielens_100k"}, separators=(",", ":")).replace('"', '""')
            f.write(
                f'{ts},U{int(row.user_id):05d},I{int(row.item_id):05d},rated,{float(row.rating)},"{meta}"\n'
            )

    _write_empty_artifacts_manifest(out_dir)
    return {
        "entities": int(len(user_mean) + len(item_mean)),
        "events": int(len(df)),
        "interactions": int(len(df)),
    }


def prepare_uci_adult(input_path: Path, out_dir: Path) -> dict[str, int]:
    """Prepare UCI Adult-like CSV into ingestion-ready files.

    Expects a CSV with an income-like target column named one of:
    - income
    - target
    - label
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)

    target_col = None
    for name in ("income", "target", "label"):
        if name in df.columns:
            target_col = name
            break
    if target_col is None:
        raise ValueError("Expected one target column among: income, target, label")

    def _to_binary(value: object) -> int:
        text = str(value).strip()
        return 1 if text in {">50K", ">50K.", "1", "true", "True", "yes", "Yes"} else 0

    labels = df[target_col].map(_to_binary).astype(int)

    numeric_cols = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 3:
        # Deterministic fallback from hashed string content.
        key_series = df.drop(columns=[target_col]).astype(str).agg("|".join, axis=1)

        def _stable_unit(seed_text: str) -> float:
            digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
            return int(digest[:12], 16) / float(16**12 - 1)

        x = key_series.map(lambda s: _stable_unit(s + "|x"))
        y = key_series.map(lambda s: _stable_unit(s + "|y"))
        z = key_series.map(lambda s: _stable_unit(s + "|z"))
    else:
        x = df[numeric_cols[0]].fillna(0.0).astype(float)
        y = df[numeric_cols[1]].fillna(0.0).astype(float)
        z = df[numeric_cols[2]].fillna(0.0).astype(float)

    # Normalize deterministic feature projections into [0,1].
    def _norm(series: pd.Series) -> pd.Series:
        min_v = float(series.min())
        max_v = float(series.max())
        if max_v - min_v <= 1e-12:
            return pd.Series([0.0] * len(series), index=series.index)
        return (series - min_v) / (max_v - min_v)

    x_n = _norm(x)
    y_n = _norm(y)
    z_n = _norm(z)

    entities_path = out_dir / "entities.csv"
    events_path = out_dir / "events.csv"
    interactions_path = out_dir / "interactions.csv"

    base_time = datetime(2025, 1, 1)

    with entities_path.open("w", encoding="utf-8") as ef, events_path.open(
        "w", encoding="utf-8"
    ) as evf, interactions_path.open("w", encoding="utf-8") as inf:
        ef.write("entity_id,attributes,created_at\n")
        evf.write("timestamp,entity_id,event_type,event_value,event_metadata\n")
        inf.write("timestamp,src_entity_id,dst_entity_id,interaction_type,interaction_value,metadata\n")

        for i in range(len(df)):
            eid = f"A{i:06d}"
            ts = (base_time).isoformat()
            target_binary = int(labels.iloc[i])
            target_reg = float(target_binary)
            target_rank = float(target_binary)

            attrs = {
                "x": float(x_n.iloc[i]),
                "y": float(y_n.iloc[i]),
                "z": float(z_n.iloc[i]),
                "target_regression": target_reg,
                "target_binary": target_binary,
                "target_ranking": target_rank,
            }
            attrs_json = json.dumps(attrs, separators=(",", ":")).replace('"', '""')
            ef.write(f'{eid},"{attrs_json}",{ts}\n')

            meta = json.dumps({"source": "uci_adult"}, separators=(",", ":")).replace('"', '""')
            evf.write(f'{ts},{eid},profile,1.0,"{meta}"\n')

            if i > 0:
                prev = f"A{i - 1:06d}"
                inf.write(f'{ts},{prev},{eid},adjacent,1.0,"{meta}"\n')

    _write_empty_artifacts_manifest(out_dir)
    return {
        "entities": int(len(df)),
        "events": int(len(df)),
        "interactions": int(max(0, len(df) - 1)),
    }
