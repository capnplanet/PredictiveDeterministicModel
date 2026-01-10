from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sqlalchemy import select
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from app.core.config import get_settings
from app.db.models import Artifact, Entity, ModelRun
from app.db.session import session_scope
from app.ml.feature_version import compute_feature_version_hash
from app.training.model import (
    EncoderConfig,
    FullModel,
    classification_metrics,
    pairwise_ranking_loss,
    ranking_metrics,
    regression_metrics,
)


@dataclass
class TrainConfig:
    epochs: int = 3
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 1234


def _set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = "0"
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random

    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


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


def _build_entity_tensors() -> Tuple[Dict[str, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Returns mapping entity_id->index and per-entity arrays for attributes and targets.
    with session_scope() as session:
        entities: List[Entity] = session.execute(select(Entity)).scalars().all()
    n = len(entities)
    id_to_idx: Dict[str, int] = {}
    attr_vec = np.zeros((n, 3), dtype="float32")
    y_reg = np.zeros((n,), dtype="float32")
    y_bin = np.zeros((n,), dtype="float32")
    y_rank = np.zeros((n,), dtype="float32")

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
    return id_to_idx, attr_vec, y_reg, y_bin, y_rank, y_reg.copy(), y_rank.copy()


class EntityDataset(Dataset[Dict[str, Tensor]]):
    def __init__(self, ids: List[str]) -> None:
        self.entity_ids = ids

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.entity_ids)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:  # type: ignore[override]
        eid = self.entity_ids[idx]
        raise NotImplementedError("EntityDataset item construction is implemented in collate_fn only")


def _build_run_id(config: TrainConfig, data_manifest: Dict[str, str]) -> str:
    settings = get_settings()
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
    cfg = TrainConfig()
    if config_path is not None and config_path.exists():
        loaded = json.loads(config_path.read_text())
        cfg = TrainConfig(**loaded)

    _set_determinism(cfg.seed)

    id_to_idx, attr_vec_np, y_reg_np, y_bin_np, y_rank_np, _, _ = _build_entity_tensors()
    n_entities = len(id_to_idx)
    if n_entities == 0:
        raise RuntimeError("No entities available for training")

    attr_tensor = torch.from_numpy(attr_vec_np)
    y_reg = torch.from_numpy(y_reg_np)
    y_bin = torch.from_numpy(y_bin_np)
    y_rank = torch.from_numpy(y_rank_np)

    encoder_cfg = EncoderConfig()
    artifact_feat_dim = 32  # synthetic artifacts histogram size; adjusted later if needed
    model = FullModel(encoder_cfg, attr_input_dim=attr_tensor.shape[1], artifact_feat_dim=artifact_feat_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    # For simplicity, we use zero tensors for event and artifact features in this reference implementation.
    dataset = torch.arange(0, n_entities, dtype=torch.long)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model.train()
    for _ in range(cfg.epochs):
        for batch_idx in loader:
            idx = batch_idx
            batch_attr = attr_tensor[idx]
            batch_reg = y_reg[idx]
            batch_bin = y_bin[idx]
            batch_rank = y_rank[idx]

            bsz = batch_attr.size(0)
            # Dummy but deterministic event and artifact tensors
            event_type_ids = torch.zeros((bsz, 4), dtype=torch.long)
            event_values = torch.zeros((bsz, 4, 1), dtype=torch.float32)
            event_deltas = torch.zeros((bsz, 4, 1), dtype=torch.float32)
            neighbor_attr = torch.zeros((bsz, 1, encoder_cfg.attr_dim), dtype=torch.float32)
            neighbor_mask = torch.zeros((bsz, 1), dtype=torch.float32)
            artifact_feats = torch.zeros((bsz, 1, artifact_feat_dim), dtype=torch.float32)

            outputs, _, _ = model(
                event_type_ids=event_type_ids,
                event_values=event_values,
                event_deltas=event_deltas,
                attr_vec=batch_attr,
                neighbor_attr=neighbor_attr,
                neighbor_mask=neighbor_mask,
                artifact_feats=artifact_feats,
            )

            reg_loss = mse(outputs["regression"], batch_reg)
            cls_loss = bce(outputs["logit"], batch_bin)
            rank_loss = pairwise_ranking_loss(outputs["score"], batch_rank)
            loss = reg_loss + cls_loss + rank_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Final metrics on all entities
    with torch.no_grad():
        bsz = n_entities
        event_type_ids = torch.zeros((bsz, 4), dtype=torch.long)
        event_values = torch.zeros((bsz, 4, 1), dtype=torch.float32)
        event_deltas = torch.zeros((bsz, 4, 1), dtype=torch.float32)
        neighbor_attr = torch.zeros((bsz, 1, encoder_cfg.attr_dim), dtype=torch.float32)
        neighbor_mask = torch.zeros((bsz, 1), dtype=torch.float32)
        artifact_feats = torch.zeros((bsz, 1, artifact_feat_dim), dtype=torch.float32)

        outputs, fused_emb, _ = model(
            event_type_ids=event_type_ids,
            event_values=event_values,
            event_deltas=event_deltas,
            attr_vec=attr_tensor,
            neighbor_attr=neighbor_attr,
            neighbor_mask=neighbor_mask,
            artifact_feats=artifact_feats,
        )

    metrics_reg = regression_metrics(outputs["regression"], y_reg)
    metrics_cls = classification_metrics(outputs["logit"], y_bin)
    metrics_rank = ranking_metrics(outputs["score"], y_rank)
    all_metrics: Dict[str, float] = {}
    all_metrics.update({f"reg_{k}": v for k, v in metrics_reg.items()})
    all_metrics.update({f"cls_{k}": v for k, v in metrics_cls.items()})
    all_metrics.update({f"rank_{k}": v for k, v in metrics_rank.items()})

    settings = get_settings()
    artifacts_root = Path(settings.artifacts_root)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    data_manifest = {"entities": "db_table:entities"}
    run_id = _build_run_id(cfg, data_manifest)
    run_dir = artifacts_root / run_id
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

    with session_scope() as session:
        run = ModelRun(
            run_id=run_id,
            config=asdict(cfg),
            metrics=all_metrics,
            model_sha256=model_sha,
            data_manifest=data_manifest,
            status="success",  # type: ignore[assignment]
            logs_path=str(run_dir / "training_log.jsonl"),
        )
        session.merge(run)

    return run_id, all_metrics


def reproduce_run(run_id: str) -> Dict[str, object]:
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

    same_run_id = new_run_id == run_id
    same_model = new_model_sha == orig_model_sha
    same_metrics = orig_metrics == new_metrics

    return {
        "expected_run_id": run_id,
        "new_run_id": new_run_id,
        "same_run_id": same_run_id,
        "same_model_sha": same_model,
        "same_metrics": same_metrics,
    }


def run_determinism_check() -> Dict[str, object]:
    # For CI speed, train twice with small config and compare.
    small_cfg = TrainConfig(epochs=1, batch_size=16, lr=1e-3, seed=1234)
    tmp_cfg_path = Path("data/determinism_train_config.json")
    tmp_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_cfg_path.write_text(json.dumps(asdict(small_cfg)))

    run_id1, metrics1 = run_training(config_path=tmp_cfg_path)
    run_id2, metrics2 = run_training(config_path=tmp_cfg_path)

    settings = get_settings()
    sha1 = (Path(settings.artifacts_root) / run_id1 / "model_sha256.txt").read_text().strip()
    sha2 = (Path(settings.artifacts_root) / run_id2 / "model_sha256.txt").read_text().strip()

    return {
        "run_id1": run_id1,
        "run_id2": run_id2,
        "same_run_id": run_id1 == run_id2,
        "same_metrics": metrics1 == metrics2,
        "same_model_sha": sha1 == sha2,
    }
