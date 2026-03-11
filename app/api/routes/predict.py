from __future__ import annotations

from collections import defaultdict
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import torch
from fastapi import APIRouter
from sqlalchemy import func, or_, select

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
from app.db.models import Artifact, Event, Interaction, ModelRun
from app.db.session import session_scope
from app.services.llm_narrative import maybe_generate_long_narrative
from app.training.model import EncoderConfig, FullModel
from app.training.train import build_entity_batch_tensors

router = APIRouter(prefix="", tags=["predict"])


def _load_latest_run_id(session) -> str:
    from pathlib import Path

    settings = get_settings()
    run_dir = Path(settings.artifacts_root)
    runs = session.query(ModelRun).order_by(ModelRun.created_at.desc()).all()
    for run in runs:
        if (run_dir / run.run_id / "model.pt").exists():
            return run.run_id
    if runs:
        raise RuntimeError("No model artifacts available for any run")
    raise RuntimeError("No model runs available")


def _load_model(run_id: str) -> Tuple[FullModel, Dict[str, float]]:
    import json
    from pathlib import Path

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


def _collect_entity_relationship_context(session, entity_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    context: Dict[str, Dict[str, Any]] = {
        entity_id: {
            "event_count": 0,
            "top_event_type": None,
            "artifact_count": 0,
            "artifact_type_counts": {},
            "strongest_related_entity": None,
            "outgoing_count": 0,
            "incoming_count": 0,
        }
        for entity_id in entity_ids
    }
    if not entity_ids:
        return context

    event_type_counts: Dict[str, Dict[str, int]] = defaultdict(dict)
    event_rows = session.execute(
        select(Event.entity_id, Event.event_type, func.count(Event.event_id))
        .where(Event.entity_id.in_(entity_ids))
        .group_by(Event.entity_id, Event.event_type)
    ).all()
    for entity_id, event_type, cnt in event_rows:
        count_int = int(cnt)
        context[entity_id]["event_count"] += count_int
        event_type_counts[entity_id][str(event_type)] = count_int
    for entity_id in entity_ids:
        if event_type_counts[entity_id]:
            context[entity_id]["top_event_type"] = sorted(
                event_type_counts[entity_id].items(), key=lambda item: (-item[1], item[0])
            )[0][0]

    artifact_rows = session.execute(
        select(Artifact.entity_id, Artifact.artifact_type, func.count(Artifact.artifact_id))
        .where(Artifact.entity_id.in_(entity_ids))
        .group_by(Artifact.entity_id, Artifact.artifact_type)
    ).all()
    for entity_id, artifact_type, cnt in artifact_rows:
        if entity_id is None:
            continue
        count_int = int(cnt)
        atype = str(artifact_type)
        context[entity_id]["artifact_count"] += count_int
        context[entity_id]["artifact_type_counts"][atype] = count_int

    strengths: Dict[str, Dict[str, float]] = {entity_id: defaultdict(float) for entity_id in entity_ids}
    interaction_rows = session.execute(
        select(Interaction.src_entity_id, Interaction.dst_entity_id, Interaction.interaction_value).where(
            or_(Interaction.src_entity_id.in_(entity_ids), Interaction.dst_entity_id.in_(entity_ids))
        )
    ).all()
    for src_entity_id, dst_entity_id, interaction_value in interaction_rows:
        weight = abs(float(interaction_value))
        if src_entity_id in context:
            context[src_entity_id]["outgoing_count"] += 1
            strengths[src_entity_id][dst_entity_id] += weight
        if dst_entity_id in context:
            context[dst_entity_id]["incoming_count"] += 1
            strengths[dst_entity_id][src_entity_id] += weight
    for entity_id in entity_ids:
        if strengths[entity_id]:
            context[entity_id]["strongest_related_entity"] = sorted(
                strengths[entity_id].items(), key=lambda item: (-item[1], item[0])
            )[0][0]

    return context


def _render_prediction_narrative(
    entity_id: str,
    rel_ctx: Dict[str, Any],
    regression: float,
    probability: float,
    ranking_score: float,
) -> str:
    event_count = int(rel_ctx.get("event_count", 0))
    top_event_type = rel_ctx.get("top_event_type") or "none"
    artifact_count = int(rel_ctx.get("artifact_count", 0))
    artifact_type_counts = rel_ctx.get("artifact_type_counts", {})
    outgoing_count = int(rel_ctx.get("outgoing_count", 0))
    incoming_count = int(rel_ctx.get("incoming_count", 0))
    strongest_related_entity: Optional[str] = rel_ctx.get("strongest_related_entity")

    artifact_parts = [
        f"{artifact_type}:{count}" for artifact_type, count in sorted(artifact_type_counts.items())
    ]
    artifact_summary = ", ".join(artifact_parts) if artifact_parts else "none"
    strongest = strongest_related_entity or "none"

    return (
        f"Entity {entity_id}: events={event_count} (top={top_event_type}); "
        f"artifacts={artifact_count} ({artifact_summary}); "
        f"relationships=outgoing:{outgoing_count}, incoming:{incoming_count}, strongest:{strongest}; "
        f"scores=regression:{regression:.4f}, probability:{probability:.4f}, ranking:{ranking_score:.4f}."
    )


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
        encoder_cfg = EncoderConfig()
        artifact_feat_dim = 32
        model_inputs, attr_tensor, entity_ids, coverage = build_entity_batch_tensors(
            entity_ids=request.entity_ids,
            encoder_cfg=encoder_cfg,
            artifact_feat_dim=artifact_feat_dim,
        )

    bsz = attr_tensor.size(0)

    with timed_performance_event(
        "predict.forward",
        run_id=run_id,
        batch_size=bsz,
    ):
        with torch.no_grad():
            outputs, fused_emb, attn = model(**model_inputs)

    with session_scope() as session:
        relationship_context = _collect_entity_relationship_context(session, entity_ids)

    preds: List[EntityPrediction] = []

    explanation_ms = 0.0
    for i, eid in enumerate(entity_ids):
        reg = float(outputs["regression"][i].item())
        prob = float(torch.sigmoid(outputs["logit"][i]).item())
        score = float(outputs["score"][i].item())
        emb_list = fused_emb[i].detach().cpu().numpy().tolist()
        template_narrative = _render_prediction_narrative(
            entity_id=eid,
            rel_ctx=relationship_context.get(eid, {}),
            regression=reg,
            probability=prob,
            ranking_score=score,
        )
        narrative_source = "template"
        narrative_long: Optional[str] = None

        if request.narrative_mode in ("llm", "both"):
            narrative_long, llm_used, _ = await maybe_generate_long_narrative(
                entity_id=eid,
                template_narrative=template_narrative,
                rel_ctx=relationship_context.get(eid, {}),
                regression=reg,
                probability=prob,
                ranking_score=score,
            )
            narrative_source = "llm" if llm_used else "template"

        if request.narrative_mode == "template":
            narrative = template_narrative
        elif request.narrative_mode == "llm":
            narrative = narrative_long or template_narrative
        else:
            narrative = narrative_long or template_narrative

        explanation = None
        if request.explanations:
            explanation_start = perf_counter()
            fused_single = fused_emb[i : i + 1]
            attr_single = attr_tensor[i : i + 1]
            artifact_single = model_inputs["artifact_feats"][i : i + 1]
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
                narrative=narrative,
                narrative_template=template_narrative,
                narrative_long=narrative_long,
                narrative_source=narrative_source,
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
        narrative_mode=request.narrative_mode,
        explanation_duration_ms=round(explanation_ms, 3),
        entities_with_events=coverage.get("entities_with_events", 0),
        entities_with_neighbors=coverage.get("entities_with_neighbors", 0),
        entities_with_artifacts=coverage.get("entities_with_artifacts", 0),
    )
    return PredictResponse(run_id=run_id, predictions=preds)
