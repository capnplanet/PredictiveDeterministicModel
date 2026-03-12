from __future__ import annotations

import asyncio
from typing import List, Literal, Tuple

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.api.routes.predict import predict as predict_entities
from app.api.schemas import PredictRequest, QueryRequest, QueryResponse, QueryResult
from app.db.models import Entity
from app.db.session import session_scope
from app.services.llm_narrative import maybe_generate_long_narrative, maybe_interpret_query

router = APIRouter(prefix="", tags=["query"])


def _extract_candidate_entity_ids(query: str, all_entity_ids: List[str], limit: int) -> Tuple[List[str], str]:
    lowered = query.lower()
    matched = [eid for eid in all_entity_ids if eid.lower() in lowered]
    if matched:
        return matched[: min(len(matched), max(limit, 25))], "explicit"

    tokens = [tok.strip().lower() for tok in lowered.replace(",", " ").split() if len(tok.strip()) >= 3]
    fuzzy = [eid for eid in all_entity_ids if any(tok in eid.lower() for tok in tokens)]
    if fuzzy:
        return fuzzy[: min(len(fuzzy), max(limit * 3, 50))], "fuzzy"

    return all_entity_ids[: min(len(all_entity_ids), max(limit * 6, 100))], "broad_scan"


def _infer_sort_intent(query: str) -> Literal["strongest", "weakest", "default"]:
    lowered = query.lower()
    weak_terms = ("weak", "weakest", "lowest", "bottom", "least")
    strong_terms = ("strong", "strongest", "highest", "top", "best")
    relationship_terms = ("relationship", "relationships", "interaction", "interactions", "connected", "connection")
    if any(term in lowered for term in weak_terms):
        return "weakest"
    if any(term in lowered for term in strong_terms):
        return "strongest"
    if any(term in lowered for term in relationship_terms):
        # Relationship-centric retrieval defaults to ranking-score ordering.
        return "strongest"
    return "default"


def _infer_probability_intent(query: str) -> Literal["elevated", "default"]:
    lowered = query.lower()
    elevated_terms = (
        "elevated probability",
        "high probability",
        "higher probability",
        "top probability",
        "likely",
        "risk",
        "risky",
    )
    if any(term in lowered for term in elevated_terms):
        return "elevated"
    return "default"


@router.post("/query", response_model=QueryResponse)
async def query_predictions(request: QueryRequest) -> QueryResponse:
    with session_scope() as session:
        all_entity_ids = list(
            session.execute(select(Entity.entity_id).order_by(Entity.created_at.desc())).scalars().all()
        )

    entity_ids, match_strategy = _extract_candidate_entity_ids(request.query, all_entity_ids, request.limit)
    sort_intent = _infer_sort_intent(request.query)
    probability_intent = _infer_probability_intent(request.query)
    interpreted_as, llm_used = await maybe_interpret_query(request.query)

    if not entity_ids:
        return QueryResponse(
            run_id=request.run_id or "",
            query=request.query,
            interpreted_as=interpreted_as,
            llm_used=llm_used,
            results=[],
        )

    try:
        pred = await predict_entities(
            PredictRequest(
                entity_ids=entity_ids,
                run_id=request.run_id,
                explanations=False,
                # Query route first scores candidates quickly, then applies
                # richer narratives only to final top-k results.
                narrative_mode="template",
            )
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "No trained model artifact is available for querying yet. "
                "Run a training operation first from Model Ops."
            ),
        ) from exc

    results = [
        QueryResult(
            entity_id=p.entity_id,
            regression=p.regression,
            probability=p.probability,
            ranking_score=p.ranking_score,
            narrative=p.narrative,
        )
        for p in pred.predictions
    ]

    if sort_intent == "strongest" and probability_intent == "elevated":
        results = sorted(results, key=lambda item: (item.ranking_score, item.probability), reverse=True)
    elif sort_intent == "weakest" and probability_intent == "elevated":
        results = sorted(results, key=lambda item: (item.ranking_score, -item.probability))
    elif sort_intent == "strongest":
        results = sorted(results, key=lambda item: item.ranking_score, reverse=True)
    elif sort_intent == "weakest":
        results = sorted(results, key=lambda item: item.ranking_score)
    elif probability_intent == "elevated":
        results = sorted(results, key=lambda item: item.probability, reverse=True)

    results = results[: request.limit]

    async def _enrich(item: QueryResult) -> Tuple[QueryResult, bool]:
        long_text, used, _ = await maybe_generate_long_narrative(
            entity_id=item.entity_id,
            template_narrative=item.narrative or "",
            rel_ctx={},
            regression=item.regression,
            probability=item.probability,
            ranking_score=item.ranking_score,
            analyst_query=request.query,
        )
        return QueryResult(
            entity_id=item.entity_id,
            regression=item.regression,
            probability=item.probability,
            ranking_score=item.ranking_score,
            narrative=long_text if long_text else item.narrative,
        ), used

    if results:
        enriched_pairs = await asyncio.gather(*[_enrich(item) for item in results])
        llm_used = llm_used or any(used for _, used in enriched_pairs)
        results = [item for item, _ in enriched_pairs]

    interpreted_with_intent = (
        f"{interpreted_as} [match={match_strategy}; order={sort_intent}; probability={probability_intent}]"
        if interpreted_as
        else f"match={match_strategy}; order={sort_intent}; probability={probability_intent}"
    )

    return QueryResponse(
        run_id=pred.run_id,
        query=request.query,
        interpreted_as=interpreted_with_intent,
        llm_used=llm_used,
        results=results,
    )
