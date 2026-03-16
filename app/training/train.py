from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import torch
from sqlalchemy import select
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from app.core.config import get_settings
from app.core.performance import emit_performance_event
from app.db.models import Artifact, Entity, Event, Interaction, ModelRun
from app.db.session import session_scope
from app.ml.feature_version import compute_feature_version_hash
from app.services.artifact_ingestion import ingest_artifacts_manifest
from app.services.csv_ingestion import (
    ingest_entities_csv,
    ingest_events_csv,
    ingest_interactions_csv,
)
from app.services.feature_extraction import extract_features_for_pending
from app.training.model import (
    EncoderConfig,
    FullModel,
    classification_metrics,
    pairwise_ranking_loss,
    ranking_metrics,
    regression_metrics,
)
from app.training.synth_data import generate_synthetic_dataset


@dataclass
class TrainConfig:
    epochs: int = 3
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 1234
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    split_strategy: str = "random"
    corpus_name: str = "default"
    threshold_policy_version: str = "v1"
    enforce_thresholds: bool = False


def _event_type_to_id(event_type: str, max_types: int = 32) -> int:
    digest = hashlib.sha256(event_type.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max_types


def _compress_feature_vector(vec: np.ndarray, out_dim: int) -> np.ndarray:
    if vec.size == out_dim:
        return vec.astype("float32")
    if vec.size == 0:
        return np.zeros((out_dim,), dtype="float32")
    chunk_edges = np.linspace(0, vec.size, num=out_dim + 1, dtype=int)
    out = np.zeros((out_dim,), dtype="float32")
    for i in range(out_dim):
        start, end = int(chunk_edges[i]), int(chunk_edges[i + 1])
        if end <= start:
            out[i] = float(vec[min(start, vec.size - 1)])
        else:
            out[i] = float(np.mean(vec[start:end]))
    return out


def _build_modality_tensors(
    id_to_idx: Dict[str, int],
    attr_vec_np: np.ndarray,
    encoder_cfg: EncoderConfig,
    artifact_feat_dim: int,
    event_seq_len: int = 4,
    neighbor_k: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
    n_entities = len(id_to_idx)
    event_type_ids = np.zeros((n_entities, event_seq_len), dtype="int64")
    event_values = np.zeros((n_entities, event_seq_len, 1), dtype="float32")
    event_deltas = np.zeros((n_entities, event_seq_len, 1), dtype="float32")
    neighbor_attr = np.zeros((n_entities, neighbor_k, encoder_cfg.attr_dim), dtype="float32")
    neighbor_mask = np.zeros((n_entities, neighbor_k), dtype="float32")
    artifact_feats = np.zeros((n_entities, 1, artifact_feat_dim), dtype="float32")

    with session_scope() as session:
        events: List[Event] = (
            session.execute(
                select(Event).order_by(
                    Event.entity_id.asc(),
                    Event.timestamp.asc(),
                    Event.event_id.asc(),
                )
            )
            .scalars()
            .all()
        )
        interactions: List[Interaction] = (
            session.execute(
                select(Interaction).order_by(
                    Interaction.src_entity_id.asc(),
                    Interaction.dst_entity_id.asc(),
                    Interaction.timestamp.asc(),
                    Interaction.interaction_id.asc(),
                )
            )
            .scalars()
            .all()
        )
        artifacts: List[Artifact] = (
            session.execute(
                select(Artifact).order_by(
                    Artifact.entity_id.asc(),
                    Artifact.artifact_id.asc(),
                )
            )
            .scalars()
            .all()
        )

    events_by_entity: Dict[str, List[Event]] = {}
    for evt in events:
        events_by_entity.setdefault(evt.entity_id, []).append(evt)

    for entity_id in sorted(events_by_entity):
        evt_list = events_by_entity[entity_id]
        idx = id_to_idx.get(entity_id)
        if idx is None:
            continue
        # Use most recent fixed-length sequence and preserve chronological order.
        sorted_events = sorted(evt_list, key=lambda e: (e.timestamp, str(e.event_id)))[-event_seq_len:]
        prev_ts: Optional[datetime] = None
        for t, evt in enumerate(sorted_events):
            event_type_ids[idx, t] = _event_type_to_id(evt.event_type)
            event_values[idx, t, 0] = float(evt.event_value)
            if prev_ts is None:
                event_deltas[idx, t, 0] = 0.0
            else:
                delta_seconds = (evt.timestamp - prev_ts).total_seconds()
                event_deltas[idx, t, 0] = float(max(0.0, delta_seconds) / 3600.0)
            prev_ts = evt.timestamp

    # Build deterministic neighbor context from strongest outgoing interactions.
    interaction_scores: Dict[str, Dict[str, float]] = {}
    for inter in interactions:
        if inter.src_entity_id not in id_to_idx or inter.dst_entity_id not in id_to_idx:
            continue
        src_scores = interaction_scores.setdefault(inter.src_entity_id, {})
        src_scores[inter.dst_entity_id] = src_scores.get(inter.dst_entity_id, 0.0) + abs(
            float(inter.interaction_value)
        )

    for src_id in sorted(interaction_scores):
        dst_scores = interaction_scores[src_id]
        src_idx = id_to_idx[src_id]
        ranked = sorted(dst_scores.items(), key=lambda item: (-item[1], item[0]))
        for k, (dst_id, _) in enumerate(ranked[:neighbor_k]):
            dst_idx = id_to_idx[dst_id]
            raw = attr_vec_np[dst_idx]
            neighbor_attr[src_idx, k, : raw.shape[0]] = raw
            neighbor_mask[src_idx, k] = 1.0

    artifact_vecs: Dict[str, List[np.ndarray]] = {}
    for art in artifacts:
        if art.entity_id is None:
            continue
        if art.entity_id not in id_to_idx:
            continue
        vec = _load_artifact_features(art)
        if vec is None:
            continue
        compressed = _compress_feature_vector(vec, artifact_feat_dim)
        artifact_vecs.setdefault(art.entity_id, []).append(compressed)

    for entity_id in sorted(artifact_vecs):
        vecs = artifact_vecs[entity_id]
        idx = id_to_idx[entity_id]
        stacked = np.stack(vecs, axis=0)
        artifact_feats[idx, 0, :] = stacked.mean(axis=0)

    coverage = {
        "entities_with_events": int(sum(1 for v in events_by_entity.values() if v)),
        "entities_with_neighbors": int((neighbor_mask.sum(axis=1) > 0).sum()),
        "entities_with_artifacts": int(sum(1 for v in artifact_vecs.values() if v)),
        "nonzero_event_values": int(np.count_nonzero(event_values)),
    }

    return (
        torch.from_numpy(event_type_ids),
        torch.from_numpy(event_values),
        torch.from_numpy(event_deltas),
        torch.from_numpy(neighbor_attr),
        torch.from_numpy(neighbor_mask),
        torch.from_numpy(artifact_feats),
        coverage,
    )


def _split_entity_indices(
    n_entities: int,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    split_strategy: str = "random",
    created_at_unix: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_entities <= 0:
        raise ValueError("n_entities must be positive")
    if val_fraction < 0.0 or test_fraction < 0.0:
        raise ValueError("val_fraction and test_fraction must be non-negative")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0")

    if n_entities < 5:
        # Keep tiny datasets train-only to avoid empty or unstable holdouts.
        all_idx = np.arange(n_entities, dtype="int64")
        return all_idx, np.array([], dtype="int64"), np.array([], dtype="int64")

    indices = np.arange(n_entities, dtype="int64")
    if split_strategy == "time":
        if created_at_unix is None or created_at_unix.shape[0] != n_entities:
            raise ValueError("created_at_unix must be provided for time split strategy")
        indices = np.argsort(created_at_unix, kind="stable").astype("int64")
    elif split_strategy == "random":
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    else:
        raise ValueError(f"Unsupported split_strategy: {split_strategy}")

    n_val = max(1, int(round(n_entities * val_fraction)))
    n_test = max(1, int(round(n_entities * test_fraction)))
    max_holdout = n_entities - 1
    if n_val + n_test > max_holdout:
        overflow = n_val + n_test - max_holdout
        reduce_test = min(overflow, n_test - 1)
        n_test -= reduce_test
        overflow -= reduce_test
        if overflow > 0:
            n_val = max(1, n_val - overflow)

    n_train = n_entities - n_val - n_test
    if n_train <= 0:
        raise ValueError("split configuration produced empty training set")

    train_idx = np.sort(indices[:n_train])
    val_idx = np.sort(indices[n_train : n_train + n_val])
    test_idx = np.sort(indices[n_train + n_val :])
    return train_idx, val_idx, test_idx


def _validate_split_integrity(
    n_entities: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Dict[str, int]:
    combined = np.concatenate([train_idx, val_idx, test_idx])
    unique_count = int(np.unique(combined).size)
    duplicates = int(combined.size - unique_count)
    missing = int(n_entities - unique_count)
    overlap = int(
        len(set(train_idx.tolist()) & set(val_idx.tolist()))
        + len(set(train_idx.tolist()) & set(test_idx.tolist()))
        + len(set(val_idx.tolist()) & set(test_idx.tolist()))
    )
    if duplicates != 0 or missing != 0 or overlap != 0:
        raise RuntimeError(
            f"Split integrity violation: duplicates={duplicates}, missing={missing}, overlap={overlap}"
        )
    return {
        "split_duplicates": duplicates,
        "split_missing": missing,
        "split_overlap": overlap,
    }


@lru_cache(maxsize=1)
def _threshold_policies() -> Dict[str, Dict[str, Dict[str, float]]]:
    policy_path = Path(__file__).with_name("threshold_policies.json")
    if not policy_path.exists():
        raise RuntimeError(f"Threshold policy config not found: {policy_path}")
    loaded = json.loads(policy_path.read_text())
    if not isinstance(loaded, dict):
        raise RuntimeError("Threshold policy config must be a JSON object")
    return cast(Dict[str, Dict[str, Dict[str, float]]], loaded)


def _resolve_thresholds(cfg: TrainConfig) -> Dict[str, float]:
    policies = _threshold_policies()
    version_policy = policies.get(cfg.threshold_policy_version)
    if version_policy is None:
        raise ValueError(f"Unknown threshold policy version: {cfg.threshold_policy_version}")
    corpus_key = cfg.corpus_name.strip().lower()
    if corpus_key in version_policy:
        return version_policy[corpus_key]
    return version_policy.get("default", {})


def _evaluate_thresholds(metrics: Dict[str, float], cfg: TrainConfig) -> Dict[str, object]:
    thresholds = _resolve_thresholds(cfg)
    violations: List[str] = []
    for metric_name, minimum in thresholds.items():
        actual = float(metrics.get(metric_name, 0.0))
        if actual < minimum:
            violations.append(f"{metric_name}={actual:.6f} < {minimum:.6f}")
    return {
        "version": cfg.threshold_policy_version,
        "corpus": cfg.corpus_name,
        "enforced": cfg.enforce_thresholds,
        "passed": len(violations) == 0,
        "thresholds": thresholds,
        "violations": violations,
    }


def _model_inputs_for_batch(
    idx: Tensor,
    batch_attr: Tensor,
    event_type_ids_all: Tensor,
    event_values_all: Tensor,
    event_deltas_all: Tensor,
    neighbor_attr_all: Tensor,
    neighbor_mask_all: Tensor,
    artifact_feats_all: Tensor,
) -> Dict[str, Tensor]:
    return {
        "event_type_ids": event_type_ids_all[idx],
        "event_values": event_values_all[idx],
        "event_deltas": event_deltas_all[idx],
        "attr_vec": batch_attr,
        "neighbor_attr": neighbor_attr_all[idx],
        "neighbor_mask": neighbor_mask_all[idx],
        "artifact_feats": artifact_feats_all[idx],
    }


def _evaluate_split(
    model: FullModel,
    indices: np.ndarray,
    attr_tensor: Tensor,
    y_reg: Tensor,
    y_bin: Tensor,
    y_rank: Tensor,
    event_type_ids_all: Tensor,
    event_values_all: Tensor,
    event_deltas_all: Tensor,
    neighbor_attr_all: Tensor,
    neighbor_mask_all: Tensor,
    artifact_feats_all: Tensor,
) -> Dict[str, float]:
    if indices.size == 0:
        return {}

    idx_tensor = torch.from_numpy(indices)
    batch_attr = attr_tensor[idx_tensor]
    batch_reg = y_reg[idx_tensor]
    batch_bin = y_bin[idx_tensor]
    batch_rank = y_rank[idx_tensor]
    inputs = _model_inputs_for_batch(
        idx=idx_tensor,
        batch_attr=batch_attr,
        event_type_ids_all=event_type_ids_all,
        event_values_all=event_values_all,
        event_deltas_all=event_deltas_all,
        neighbor_attr_all=neighbor_attr_all,
        neighbor_mask_all=neighbor_mask_all,
        artifact_feats_all=artifact_feats_all,
    )

    with torch.no_grad():
        outputs, _, _ = model(**inputs)

    metrics_reg = regression_metrics(outputs["regression"], batch_reg)
    metrics_cls = classification_metrics(outputs["logit"], batch_bin)
    metrics_rank = ranking_metrics(outputs["score"], batch_rank)

    metrics: Dict[str, float] = {}
    metrics.update({f"reg_{k}": v for k, v in metrics_reg.items()})
    metrics.update({f"cls_{k}": v for k, v in metrics_cls.items()})
    metrics.update({f"rank_{k}": v for k, v in metrics_rank.items()})
    return metrics


def _set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = "0"
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random

    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # If threads were already configured by a prior run in this process, keep existing value.
        pass


def _load_artifact_features(artifact: Artifact) -> Optional[np.ndarray]:
    if not artifact.feature_dim or not artifact.feature_version_hash:
        return None
    settings = get_settings()
    cache_path = Path(settings.data_root) / "feature_cache" / f"{artifact.sha256}.npy"
    if not cache_path.exists():
        return None
    raw = np.fromfile(cache_path, dtype="float32")
    if raw.size != artifact.feature_dim:
        return None
    return raw


def _build_entity_tensors() -> Tuple[
    Dict[str, int],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    # Returns mapping entity_id->index and per-entity arrays for attributes and targets.
    with session_scope() as session:
        entities: List[Entity] = (
            session.execute(select(Entity).order_by(Entity.created_at.asc(), Entity.entity_id.asc()))
            .scalars()
            .all()
        )
    n = len(entities)
    id_to_idx: Dict[str, int] = {}
    attr_vec = np.zeros((n, 3), dtype="float32")
    y_reg = np.zeros((n,), dtype="float32")
    y_bin = np.zeros((n,), dtype="float32")
    y_rank = np.zeros((n,), dtype="float32")
    created_unix = np.zeros((n,), dtype="float64")

    for i, ent in enumerate(entities):
        id_to_idx[ent.entity_id] = i
        attrs = ent.attributes or {}
        x = float(attrs.get("x", 0.0))
        y = float(attrs.get("y", 0.0))
        z = float(attrs.get("z", 0.0))
        attr_vec[i] = np.array([x, y, z], dtype="float32")
        y_reg[i] = float(attrs.get("target_regression", 0.0))
        y_bin[i] = float(attrs.get("target_binary", 0.0))
        y_rank[i] = float(attrs.get("target_ranking", 0.0))
        created_unix[i] = ent.created_at.timestamp()
    return id_to_idx, attr_vec, y_reg, y_bin, y_rank, y_reg.copy(), y_rank.copy(), created_unix


def build_entity_batch_tensors(
    entity_ids: List[str],
    encoder_cfg: EncoderConfig,
    artifact_feat_dim: int,
) -> Tuple[Dict[str, Tensor], Tensor, List[str], Dict[str, int]]:
    with session_scope() as session:
        rows: List[Entity] = (
            session.execute(
                select(Entity)
                .where(Entity.entity_id.in_(entity_ids))
                .order_by(Entity.entity_id.asc())
            )
            .scalars()
            .all()
        )

    if not rows:
        raise RuntimeError("No matching entities found")

    # Preserve request order deterministically while skipping unknown IDs.
    row_map = {row.entity_id: row for row in rows}
    ordered = [row_map[eid] for eid in entity_ids if eid in row_map]
    if not ordered:
        raise RuntimeError("No matching entities found")

    id_to_idx: Dict[str, int] = {}
    attr_vec = np.zeros((len(ordered), 3), dtype="float32")
    resolved_ids: List[str] = []
    for i, row in enumerate(ordered):
        attrs = row.attributes or {}
        x = float(attrs.get("x", 0.0))
        y = float(attrs.get("y", 0.0))
        z = float(attrs.get("z", 0.0))
        attr_vec[i] = np.array([x, y, z], dtype="float32")
        id_to_idx[row.entity_id] = i
        resolved_ids.append(row.entity_id)

    (
        event_type_ids_all,
        event_values_all,
        event_deltas_all,
        neighbor_attr_all,
        neighbor_mask_all,
        artifact_feats_all,
        coverage,
    ) = _build_modality_tensors(
        id_to_idx=id_to_idx,
        attr_vec_np=attr_vec,
        encoder_cfg=encoder_cfg,
        artifact_feat_dim=artifact_feat_dim,
    )

    idx = torch.arange(0, len(resolved_ids), dtype=torch.long)
    attr_tensor = torch.from_numpy(attr_vec)
    inputs = _model_inputs_for_batch(
        idx=idx,
        batch_attr=attr_tensor,
        event_type_ids_all=event_type_ids_all,
        event_values_all=event_values_all,
        event_deltas_all=event_deltas_all,
        neighbor_attr_all=neighbor_attr_all,
        neighbor_mask_all=neighbor_mask_all,
        artifact_feats_all=artifact_feats_all,
    )
    return inputs, attr_tensor, resolved_ids, coverage


class EntityDataset(Dataset[Dict[str, Tensor]]):
    def __init__(self, ids: List[str]) -> None:
        self.entity_ids = ids

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.entity_ids)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:  # type: ignore[override]
        raise NotImplementedError("EntityDataset items are built in collate_fn only")


def _build_run_id(config: TrainConfig, data_manifest: Dict[str, object]) -> str:
    # Hash of config + data manifest + feature version hash + repo manifest (simplified).
    feature_hash = compute_feature_version_hash()
    repo_manifest = {"placeholder": "code_hash"}
    payload = {
        "config": asdict(config),
        "data_manifest": data_manifest,
        "feature_version_hash": feature_hash,
        "repo_manifest": repo_manifest,
    }
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def run_training(config_path: Optional[Path] = None) -> Tuple[str, Dict[str, float]]:
    training_started = perf_counter()
    cfg = TrainConfig()
    if config_path is not None and config_path.exists():
        loaded = json.loads(config_path.read_text())
        cfg = TrainConfig(**loaded)
    emit_performance_event(
        "training.config_loaded",
        config_path=str(config_path) if config_path is not None else None,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        seed=cfg.seed,
        val_fraction=cfg.val_fraction,
        test_fraction=cfg.test_fraction,
        split_strategy=cfg.split_strategy,
        corpus_name=cfg.corpus_name,
        threshold_policy_version=cfg.threshold_policy_version,
        enforce_thresholds=cfg.enforce_thresholds,
    )

    _set_determinism(cfg.seed)

    prep_started = perf_counter()
    id_to_idx, attr_vec_np, y_reg_np, y_bin_np, y_rank_np, _, _, created_unix = _build_entity_tensors()
    n_entities = len(id_to_idx)
    if n_entities == 0:
        raise RuntimeError("No entities available for training")
    emit_performance_event(
        "training.data_prepared",
        duration_ms=(perf_counter() - prep_started) * 1000.0,
        entities=n_entities,
    )

    train_indices, val_indices, test_indices = _split_entity_indices(
        n_entities=n_entities,
        seed=cfg.seed,
        val_fraction=cfg.val_fraction,
        test_fraction=cfg.test_fraction,
        split_strategy=cfg.split_strategy,
        created_at_unix=created_unix,
    )
    split_integrity = _validate_split_integrity(
        n_entities=n_entities,
        train_idx=train_indices,
        val_idx=val_indices,
        test_idx=test_indices,
    )
    emit_performance_event(
        "training.data_split",
        train_size=int(train_indices.size),
        val_size=int(val_indices.size),
        test_size=int(test_indices.size),
        entities=n_entities,
        split_strategy=cfg.split_strategy,
        **split_integrity,
    )

    attr_tensor = torch.from_numpy(attr_vec_np)
    y_reg = torch.from_numpy(y_reg_np)
    y_bin = torch.from_numpy(y_bin_np)
    y_rank = torch.from_numpy(y_rank_np)

    encoder_cfg = EncoderConfig()
    artifact_feat_dim = 32  # synthetic artifacts histogram size; adjusted later if needed
    (
        event_type_ids_all,
        event_values_all,
        event_deltas_all,
        neighbor_attr_all,
        neighbor_mask_all,
        artifact_feats_all,
        modality_coverage,
    ) = _build_modality_tensors(
        id_to_idx=id_to_idx,
        attr_vec_np=attr_vec_np,
        encoder_cfg=encoder_cfg,
        artifact_feat_dim=artifact_feat_dim,
    )
    emit_performance_event(
        "training.modalities",
        **modality_coverage,
    )

    model = FullModel(
        encoder_cfg,
        attr_input_dim=attr_tensor.shape[1],
        artifact_feat_dim=artifact_feat_dim,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    # For simplicity, we use zero tensors for event and artifact features in this reference implementation.
    dataset = torch.from_numpy(train_indices)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model.train()
    train_loop_started = perf_counter()
    for _ in range(cfg.epochs):
        for batch_idx in loader:
            idx = batch_idx
            batch_attr = attr_tensor[idx]
            batch_reg = y_reg[idx]
            batch_bin = y_bin[idx]
            batch_rank = y_rank[idx]

            inputs = _model_inputs_for_batch(
                idx=idx,
                batch_attr=batch_attr,
                event_type_ids_all=event_type_ids_all,
                event_values_all=event_values_all,
                event_deltas_all=event_deltas_all,
                neighbor_attr_all=neighbor_attr_all,
                neighbor_mask_all=neighbor_mask_all,
                artifact_feats_all=artifact_feats_all,
            )
            outputs, _, _ = model(**inputs)

            reg_loss = mse(outputs["regression"], batch_reg)
            cls_loss = bce(outputs["logit"], batch_bin)
            rank_loss = pairwise_ranking_loss(outputs["score"], batch_rank)
            loss = reg_loss + cls_loss + rank_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    emit_performance_event(
        "training.train_loop",
        duration_ms=(perf_counter() - train_loop_started) * 1000.0,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        entities=n_entities,
    )

    # Final metrics on deterministic train/val/test splits (holdout-first)
    eval_started = perf_counter()
    train_metrics = _evaluate_split(
        model=model,
        indices=train_indices,
        attr_tensor=attr_tensor,
        y_reg=y_reg,
        y_bin=y_bin,
        y_rank=y_rank,
        event_type_ids_all=event_type_ids_all,
        event_values_all=event_values_all,
        event_deltas_all=event_deltas_all,
        neighbor_attr_all=neighbor_attr_all,
        neighbor_mask_all=neighbor_mask_all,
        artifact_feats_all=artifact_feats_all,
    )
    val_metrics = _evaluate_split(
        model=model,
        indices=val_indices,
        attr_tensor=attr_tensor,
        y_reg=y_reg,
        y_bin=y_bin,
        y_rank=y_rank,
        event_type_ids_all=event_type_ids_all,
        event_values_all=event_values_all,
        event_deltas_all=event_deltas_all,
        neighbor_attr_all=neighbor_attr_all,
        neighbor_mask_all=neighbor_mask_all,
        artifact_feats_all=artifact_feats_all,
    )
    test_metrics = _evaluate_split(
        model=model,
        indices=test_indices,
        attr_tensor=attr_tensor,
        y_reg=y_reg,
        y_bin=y_bin,
        y_rank=y_rank,
        event_type_ids_all=event_type_ids_all,
        event_values_all=event_values_all,
        event_deltas_all=event_deltas_all,
        neighbor_attr_all=neighbor_attr_all,
        neighbor_mask_all=neighbor_mask_all,
        artifact_feats_all=artifact_feats_all,
    )

    # Keep API compatibility by exposing unsuffixed metrics as primary holdout metrics.
    primary_metrics = test_metrics if test_metrics else train_metrics
    all_metrics: Dict[str, float] = dict(primary_metrics)
    all_metrics.update({f"train_{k}": v for k, v in train_metrics.items()})
    all_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
    all_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
    all_metrics["split_train_size"] = float(train_indices.size)
    all_metrics["split_val_size"] = float(val_indices.size)
    all_metrics["split_test_size"] = float(test_indices.size)

    threshold_result = _evaluate_thresholds(primary_metrics, cfg)
    all_metrics["thresholds_passed"] = 1.0 if bool(threshold_result["passed"]) else 0.0

    emit_performance_event(
        "training.evaluation",
        duration_ms=(perf_counter() - eval_started) * 1000.0,
        metric_count=len(all_metrics),
        primary_split="test" if test_metrics else "train",
    )
    emit_performance_event(
        "training.thresholds",
        policy_version=str(threshold_result["version"]),
        corpus=str(threshold_result["corpus"]),
        enforced=bool(threshold_result["enforced"]),
        passed=bool(threshold_result["passed"]),
        violation_count=len(cast(List[str], threshold_result["violations"])),
    )

    if cfg.enforce_thresholds and not bool(threshold_result["passed"]):
        joined = "; ".join(cast(List[str], threshold_result["violations"]))
        raise RuntimeError(
            "Threshold policy failed "
            f"(version={cfg.threshold_policy_version}, corpus={cfg.corpus_name}): {joined}"
        )

    settings = get_settings()
    artifacts_root = Path(settings.artifacts_root)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    data_manifest = {
        "entities": "db_table:entities",
        "split": {
            "train_size": int(train_indices.size),
            "val_size": int(val_indices.size),
            "test_size": int(test_indices.size),
            "seed": cfg.seed,
            "val_fraction": cfg.val_fraction,
            "test_fraction": cfg.test_fraction,
            "split_strategy": cfg.split_strategy,
        },
        "threshold_policy": threshold_result,
    }
    run_id = _build_run_id(cfg, data_manifest)
    run_dir = artifacts_root / run_id
    logs_path = str(run_dir / "training_log.jsonl")

    # Persist a pending run record first so any partial artifact write can be
    # detected and recovered without ambiguous state.
    with session_scope() as session:
        session.merge(
            ModelRun(
                run_id=run_id,
                config=asdict(cfg),
                metrics={},
                model_sha256="pending",
                data_manifest=data_manifest,
                status="pending",  # type: ignore[assignment]
                logs_path=logs_path,
            )
        )

    persist_started = perf_counter()
    try:
        run_dir.mkdir(parents=True, exist_ok=True)

        model_path = run_dir / "model.pt"
        torch.save({"state_dict": model.state_dict(), "config": asdict(cfg)}, model_path)
        model_bytes = model_path.read_bytes()
        model_sha = hashlib.sha256(model_bytes).hexdigest()

        (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
        (run_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
        (run_dir / "data_manifest.json").write_text(json.dumps(data_manifest, indent=2))
        (run_dir / "training_log.jsonl").write_text("training completed\n")
        (run_dir / "model_sha256.txt").write_text(model_sha)
        emit_performance_event(
            "training.persist_artifacts",
            duration_ms=(perf_counter() - persist_started) * 1000.0,
            run_id=run_id,
            run_dir=str(run_dir),
        )

        db_started = perf_counter()
        with session_scope() as session:
            run = session.get(ModelRun, run_id)
            if run is None:
                run = ModelRun(
                    run_id=run_id,
                    config=asdict(cfg),
                    metrics=all_metrics,
                    model_sha256=model_sha,
                    data_manifest=data_manifest,
                    status="success",  # type: ignore[assignment]
                    logs_path=logs_path,
                )
            else:
                run.config = asdict(cfg)
                run.metrics = all_metrics
                run.model_sha256 = model_sha
                run.data_manifest = data_manifest
                run.status = "success"  # type: ignore[assignment]
                run.logs_path = logs_path
            session.merge(run)
        emit_performance_event(
            "training.persist_db",
            duration_ms=(perf_counter() - db_started) * 1000.0,
            run_id=run_id,
        )
    except Exception as exc:
        failure_manifest = dict(data_manifest)
        failure_manifest["persistence_error"] = f"{type(exc).__name__}: {exc}"
        with session_scope() as session:
            run = session.get(ModelRun, run_id)
            if run is None:
                run = ModelRun(
                    run_id=run_id,
                    config=asdict(cfg),
                    metrics={"persistence_failed": 1.0},
                    model_sha256="failed",
                    data_manifest=failure_manifest,
                    status="failed",  # type: ignore[assignment]
                    logs_path=logs_path,
                )
            else:
                run.metrics = {"persistence_failed": 1.0}
                run.model_sha256 = "failed"
                run.data_manifest = failure_manifest
                run.status = "failed"  # type: ignore[assignment]
                run.logs_path = logs_path
            session.merge(run)
        emit_performance_event(
            "training.persist_failed",
            run_id=run_id,
            error_type=type(exc).__name__,
        )
        raise

    emit_performance_event(
        "training.total",
        duration_ms=(perf_counter() - training_started) * 1000.0,
        run_id=run_id,
        entities=n_entities,
        epochs=cfg.epochs,
    )

    return run_id, all_metrics


def reproduce_run(run_id: str) -> Dict[str, object]:
    started = perf_counter()
    settings = get_settings()
    run_dir = Path(settings.artifacts_root) / run_id
    if not run_dir.exists():
        raise RuntimeError(f"Run directory not found for {run_id}")

    orig_metrics = json.loads((run_dir / "metrics.json").read_text())
    orig_cfg_dict = json.loads((run_dir / "config.json").read_text())
    orig_model_sha = (run_dir / "model_sha256.txt").read_text().strip()

    cfg_path = run_dir / "config.json"
    new_run_id, new_metrics = run_training(config_path=cfg_path)

    new_model_sha = (Path(settings.artifacts_root) / new_run_id / "model_sha256.txt").read_text().strip()

    # Compare predictions on a deterministic subset of entities
    _set_determinism(int(orig_cfg_dict.get("seed", 1234)))
    id_to_idx, attr_vec_np, y_reg_np, y_bin_np, y_rank_np, _, _, _ = _build_entity_tensors()
    n_entities = len(id_to_idx)
    subset = min(16, n_entities)
    if subset == 0:
        raise RuntimeError("No entities available for reproducibility check")
    indices = np.arange(subset, dtype="int64")
    attr_tensor = torch.from_numpy(attr_vec_np)[indices]

    encoder_cfg = EncoderConfig()
    artifact_feat_dim = 32
    model1 = FullModel(
        encoder_cfg,
        attr_input_dim=attr_tensor.shape[1],
        artifact_feat_dim=artifact_feat_dim,
    )
    model2 = FullModel(
        encoder_cfg,
        attr_input_dim=attr_tensor.shape[1],
        artifact_feat_dim=artifact_feat_dim,
    )

    def _load_state(target_model: FullModel, rid: str) -> None:
        settings_local = get_settings()
        run_dir_local = Path(settings_local.artifacts_root) / rid
        data = torch.load(run_dir_local / "model.pt", map_location="cpu")
        target_model.load_state_dict(data["state_dict"])
        target_model.eval()

    _load_state(model1, run_id)
    _load_state(model2, new_run_id)

    def _predict(model_inst: FullModel, attr: torch.Tensor) -> torch.Tensor:
        bsz = attr.size(0)
        event_type_ids = torch.zeros((bsz, 4), dtype=torch.long)
        event_values = torch.zeros((bsz, 4, 1), dtype=torch.float32)
        event_deltas = torch.zeros((bsz, 4, 1), dtype=torch.float32)
        neighbor_attr = torch.zeros((bsz, 1, encoder_cfg.attr_dim), dtype=torch.float32)
        neighbor_mask = torch.zeros((bsz, 1), dtype=torch.float32)
        artifact_feats = torch.zeros((bsz, 1, artifact_feat_dim), dtype=torch.float32)
        with torch.no_grad():
            outputs, _, _ = model_inst(
                event_type_ids=event_type_ids,
                event_values=event_values,
                event_deltas=event_deltas,
                attr_vec=attr,
                neighbor_attr=neighbor_attr,
                neighbor_mask=neighbor_mask,
                artifact_feats=artifact_feats,
            )
        return outputs["regression"]

    preds1 = _predict(model1, attr_tensor).detach().cpu().numpy()
    preds2 = _predict(model2, attr_tensor).detach().cpu().numpy()

    same_run_id = new_run_id == run_id
    same_model = new_model_sha == orig_model_sha
    same_metrics = orig_metrics == new_metrics
    same_predictions = bool(np.array_equal(preds1, preds2))

    report = {
        "expected_run_id": run_id,
        "new_run_id": new_run_id,
        "same_run_id": same_run_id,
        "same_model_sha": same_model,
        "same_metrics": same_metrics,
        "same_predictions": same_predictions,
    }
    emit_performance_event(
        "training.reproduce",
        duration_ms=(perf_counter() - started) * 1000.0,
        expected_run_id=run_id,
        same_run_id=same_run_id,
        same_model_sha=same_model,
        same_metrics=same_metrics,
        same_predictions=same_predictions,
    )
    return report


def run_determinism_check() -> Dict[str, object]:
    """End-to-end determinism check on a synthetic dataset.

    Steps:
    - Generate a small synthetic dataset (CSV + artifacts).
    - Ingest CSVs and artifact manifest into Postgres.
    - Extract features.
    - Train twice with the same config.
    - Compare run_id, metrics, model hash, and predictions.
    """

    determinism_started = perf_counter()
    # Reset state (filesystem + database) to make this check idempotent
    settings = get_settings()
    synth_dir = Path("data/determinism_synth")
    artifacts_store = Path(settings.data_root) / "artifacts_store"
    feature_cache = Path(settings.data_root) / "feature_cache"

    for path in [synth_dir, artifacts_store, feature_cache]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    reset_started = perf_counter()
    with session_scope() as session:
        session.query(Artifact).delete()
        session.query(ModelRun).delete()
        session.query(Interaction).delete()
        session.query(Event).delete()
        session.query(Entity).delete()
    emit_performance_event(
        "determinism.reset_state",
        duration_ms=(perf_counter() - reset_started) * 1000.0,
    )

    # Generate synthetic data
    gen_started = perf_counter()
    generate_synthetic_dataset(
        out_dir=synth_dir,
        n_entities=64,
        n_events=512,
        n_interactions=256,
        n_artifacts=64,
        seed=2024,
    )
    emit_performance_event(
        "determinism.generate_synthetic",
        duration_ms=(perf_counter() - gen_started) * 1000.0,
        entities=64,
        events=512,
        interactions=256,
        artifacts=64,
    )

    # Ingest into DB
    ingest_started = perf_counter()
    with session_scope() as session:
        ingest_entities_csv(session, synth_dir / "entities.csv")
        ingest_events_csv(session, synth_dir / "events.csv")
        ingest_interactions_csv(session, synth_dir / "interactions.csv")
    emit_performance_event(
        "determinism.ingest_core",
        duration_ms=(perf_counter() - ingest_started) * 1000.0,
    )

    # Artifact ingest + feature extraction
    feature_started = perf_counter()
    with session_scope() as session:
        ingest_artifacts_manifest(session, synth_dir / "artifacts_manifest.csv")
        extract_features_for_pending(session)
    emit_performance_event(
        "determinism.artifacts_and_features",
        duration_ms=(perf_counter() - feature_started) * 1000.0,
    )

    # Training with fixed config
    small_cfg = TrainConfig(epochs=1, batch_size=16, lr=1e-3, seed=1234)
    tmp_cfg_path = Path("data/determinism_train_config.json")
    tmp_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_cfg_path.write_text(json.dumps(asdict(small_cfg)))

    train_started = perf_counter()
    run_id1, metrics1 = run_training(config_path=tmp_cfg_path)
    run_id2, metrics2 = run_training(config_path=tmp_cfg_path)
    train_duration_ms = (perf_counter() - train_started) * 1000.0

    sha1 = (Path(settings.artifacts_root) / run_id1 / "model_sha256.txt").read_text().strip()
    sha2 = (Path(settings.artifacts_root) / run_id2 / "model_sha256.txt").read_text().strip()

    # Compare predictions on a deterministic subset
    id_to_idx, attr_vec_np, _, _, _, _, _, _ = _build_entity_tensors()
    n_entities = len(id_to_idx)
    subset = min(16, n_entities)
    indices = np.arange(subset, dtype="int64")
    attr_tensor = torch.from_numpy(attr_vec_np)[indices]

    encoder_cfg = EncoderConfig()
    artifact_feat_dim = 32
    model1 = FullModel(
        encoder_cfg,
        attr_input_dim=attr_tensor.shape[1],
        artifact_feat_dim=artifact_feat_dim,
    )
    model2 = FullModel(
        encoder_cfg,
        attr_input_dim=attr_tensor.shape[1],
        artifact_feat_dim=artifact_feat_dim,
    )

    def _load_state(run_id_local: str, model_inst: FullModel) -> None:
        run_dir_local = Path(settings.artifacts_root) / run_id_local
        data = torch.load(run_dir_local / "model.pt", map_location="cpu")
        model_inst.load_state_dict(data["state_dict"])
        model_inst.eval()

    _load_state(run_id1, model1)
    _load_state(run_id2, model2)

    def _predict(model_inst: FullModel) -> np.ndarray:
        bsz = attr_tensor.size(0)
        event_type_ids = torch.zeros((bsz, 4), dtype=torch.long)
        event_values = torch.zeros((bsz, 4, 1), dtype=torch.float32)
        event_deltas = torch.zeros((bsz, 4, 1), dtype=torch.float32)
        neighbor_attr = torch.zeros((bsz, 1, encoder_cfg.attr_dim), dtype=torch.float32)
        neighbor_mask = torch.zeros((bsz, 1), dtype=torch.float32)
        artifact_feats = torch.zeros((bsz, 1, artifact_feat_dim), dtype=torch.float32)
        with torch.no_grad():
            outputs, _, _ = model_inst(
                event_type_ids=event_type_ids,
                event_values=event_values,
                event_deltas=event_deltas,
                attr_vec=attr_tensor,
                neighbor_attr=neighbor_attr,
                neighbor_mask=neighbor_mask,
                artifact_feats=artifact_feats,
            )
        return outputs["regression"].detach().cpu().numpy()

    compare_started = perf_counter()
    preds1 = _predict(model1)
    preds2 = _predict(model2)
    compare_duration_ms = (perf_counter() - compare_started) * 1000.0

    report = {
        "run_id1": run_id1,
        "run_id2": run_id2,
        "same_run_id": run_id1 == run_id2,
        "same_metrics": metrics1 == metrics2,
        "same_model_sha": sha1 == sha2,
        "same_predictions": bool(np.array_equal(preds1, preds2)),
    }
    emit_performance_event(
        "determinism.train_twice",
        duration_ms=train_duration_ms,
        run_id1=run_id1,
        run_id2=run_id2,
    )
    emit_performance_event(
        "determinism.compare_predictions",
        duration_ms=compare_duration_ms,
        same_predictions=report["same_predictions"],
    )
    emit_performance_event(
        "determinism.total",
        duration_ms=(perf_counter() - determinism_started) * 1000.0,
        same_run_id=report["same_run_id"],
        same_metrics=report["same_metrics"],
        same_model_sha=report["same_model_sha"],
        same_predictions=report["same_predictions"],
    )
    return report
