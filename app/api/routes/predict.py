from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from fastapi import APIRouter
from sqlalchemy import select
from time import perf_counter

from app.api.schemas import (
    ArtifactAttribution,
    AttentionExplanation,
    EntityExplanation,
    EntityPrediction,
    PredictRequest,
    PredictResponse,
)
from app.core.config import get_settings
from app.core.performance import emit_performance_event, timed_performance_event
from app.db.models import Entity, ModelRun
from app.db.session import session_scope
from app.training.model import EncoderConfig, FullModel

router = APIRouter(prefix="", tags=["predict"])


def _load_latest_run_id(session) -> str:
    run = (
        session.query(ModelRun)
        .order_by(ModelRun.created_at.desc())
        .limit(1)
        .one_or_none()
    )
    if run is None:
        raise RuntimeError("No model runs available")
    return run.run_id


def _load_model(run_id: str) -> Tuple[FullModel, Dict[str, float]]:
    from pathlib import Path
    import json

    settings = get_settings()
    run_dir = Path(settings.artifacts_root)

    path = run_dir / run_id / "model.pt"
    if not path.exists():
        raise RuntimeError(f"Model file not found for run {run_id}")
    data = torch.load(path, map_location="cpu")
    encoder_cfg = EncoderConfig()
    model = FullModel(encoder_cfg, attr_input_dim=3, artifact_feat_dim=32)
    model.load_state_dict(data["state_dict"])
    model.eval()
    metrics_path = run_dir / run_id / "metrics.json"
    metrics: Dict[str, float] = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
    return model, metrics


def _build_entity_batch(entity_ids: List[str]) -> Tuple[torch.Tensor, List[str]]:
    with session_scope() as session:
        entities: List[Entity] = (
            session.execute(select(Entity).where(Entity.entity_id.in_(entity_ids))).scalars().all()
        )
    if not entities:
        raise RuntimeError("No matching entities found")
    ids: List[str] = []
    attrs: List[List[float]] = []
    for e in entities:
        attrs_dict = e.attributes or {}
        x = float(attrs_dict.get("x", 0.0))
        y = float(attrs_dict.get("y", 0.0))
        z = float(attrs_dict.get("z", 0.0))
        ids.append(e.entity_id)
        attrs.append([x, y, z])
    attr_tensor = torch.tensor(attrs, dtype=torch.float32)
    return attr_tensor, ids


def _integrated_gradients_fused(model: FullModel, fused: torch.Tensor, target: torch.Tensor) -> List[float]:
    baseline = torch.zeros_like(fused)
    steps = 16
    total_grad = torch.zeros_like(fused)
    for alpha in torch.linspace(0.0, 1.0, steps):
        emb = baseline + alpha * (fused - baseline)
        emb.requires_grad_(True)
        outputs = model.head(emb)
        scalar = outputs["regression"].sum()
        model.zero_grad()
        if emb.grad is not None:
            emb.grad.zero_()
        scalar.backward()
        assert emb.grad is not None
        total_grad = total_grad + emb.grad
    avg_grad = total_grad / float(steps)
    attributions = (fused - baseline) * avg_grad
    return attributions.detach().cpu().numpy()[0].tolist()


def _artifact_attributions(
    model: FullModel,
    attr_vec: torch.Tensor,
    artifact_feats: torch.Tensor,
) -> List[float]:
    baseline = torch.zeros_like(artifact_feats)
    steps = 8
    total_grad = torch.zeros_like(artifact_feats)
    for alpha in torch.linspace(0.0, 1.0, steps):
        feats = baseline + alpha * (artifact_feats - baseline)
        feats.requires_grad_(True)
        bsz = attr_vec.size(0)
        event_type_ids = torch.zeros((bsz, 4), dtype=torch.long)
        event_values = torch.zeros((bsz, 4, 1), dtype=torch.float32)
        event_deltas = torch.zeros((bsz, 4, 1), dtype=torch.float32)
        neighbor_attr = torch.zeros((bsz, 1, 16), dtype=torch.float32)
        neighbor_mask = torch.zeros((bsz, 1), dtype=torch.float32)
        outputs, _, _ = model(
            event_type_ids=event_type_ids,
            event_values=event_values,
            event_deltas=event_deltas,
            attr_vec=attr_vec,
            neighbor_attr=neighbor_attr,
            neighbor_mask=neighbor_mask,
            artifact_feats=feats,
        )
        scalar = outputs["regression"].sum()
        model.zero_grad()
        if feats.grad is not None:
            feats.grad.zero_()
        scalar.backward()
        assert feats.grad is not None
        total_grad = total_grad + feats.grad
    avg_grad = total_grad / float(steps)
    attrs = (artifact_feats - baseline) * avg_grad
    # Sum over feature dimensions for a single artifact per entity
    contrib = attrs.detach().cpu().numpy().sum(axis=2)[0].tolist()
    return contrib


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    predict_start = perf_counter()
    with session_scope() as session:
        run_id = request.run_id or _load_latest_run_id(session)

    with timed_performance_event(
        "predict.model_load",
        run_id=run_id,
    ):
        model, _ = _load_model(run_id)

    with timed_performance_event(
        "predict.build_batch",
        run_id=run_id,
        requested_entities=len(request.entity_ids),
    ):
        attr_tensor, entity_ids = _build_entity_batch(request.entity_ids)

    bsz = attr_tensor.size(0)
    event_type_ids = torch.zeros((bsz, 4), dtype=torch.long)
    event_values = torch.zeros((bsz, 4, 1), dtype=torch.float32)
    event_deltas = torch.zeros((bsz, 4, 1), dtype=torch.float32)
    neighbor_attr = torch.zeros((bsz, 1, 16), dtype=torch.float32)
    neighbor_mask = torch.zeros((bsz, 1), dtype=torch.float32)
    artifact_feats = torch.zeros((bsz, 1, 32), dtype=torch.float32)

    with timed_performance_event(
        "predict.forward",
        run_id=run_id,
        batch_size=bsz,
    ):
        with torch.no_grad():
            outputs, fused_emb, attn = model(
                event_type_ids=event_type_ids,
                event_values=event_values,
                event_deltas=event_deltas,
                attr_vec=attr_tensor,
                neighbor_attr=neighbor_attr,
                neighbor_mask=neighbor_mask,
                artifact_feats=artifact_feats,
            )

    preds: List[EntityPrediction] = []

    explanation_ms = 0.0
    for i, eid in enumerate(entity_ids):
        reg = float(outputs["regression"][i].item())
        prob = float(torch.sigmoid(outputs["logit"][i]).item())
        score = float(outputs["score"][i].item())
        emb_list = fused_emb[i].detach().cpu().numpy().tolist()

        explanation = None
        if request.explanations:
            explanation_start = perf_counter()
            fused_single = fused_emb[i : i + 1]
            attr_single = attr_tensor[i : i + 1]
            artifact_single = artifact_feats[i : i + 1]
            fused_attr = _integrated_gradients_fused(model, fused_single, outputs["regression"][i : i + 1])
            attn_weights = attn[i].detach().cpu().numpy().tolist()
            art_contrib = _artifact_attributions(model, attr_single, artifact_single)
            # Map artifact contributions to synthetic single-artifact slots
            art_attr = [
                ArtifactAttribution(sha256=f"synthetic_{idx}", contribution=val)
                for idx, val in enumerate(art_contrib)
            ]
            explanation = EntityExplanation(
                fused_attribution=fused_attr,
                attention=AttentionExplanation(token_weights=attn_weights),
                artifact_attributions=art_attr,
            )
            explanation_ms += (perf_counter() - explanation_start) * 1000.0

        preds.append(
            EntityPrediction(
                entity_id=eid,
                regression=reg,
                probability=prob,
                ranking_score=score,
                embedding=emb_list,
                explanation=explanation,
            )
        )

    emit_performance_event(
        "api.predict",
        status="ok",
        duration_ms=(perf_counter() - predict_start) * 1000.0,
        run_id=run_id,
        batch_size=len(entity_ids),
        explanations=request.explanations,
        explanation_duration_ms=round(explanation_ms, 3),
    )
    return PredictResponse(run_id=run_id, predictions=preds)
