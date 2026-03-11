from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import httpx

from app.core.config import get_settings
from app.core.performance import emit_performance_event


def _extract_generated_text(payload: Any) -> Optional[str]:
    if isinstance(payload, list) and payload:
        item = payload[0]
        if isinstance(item, dict):
            text = item.get("generated_text")
            if isinstance(text, str):
                return text.strip()
    if isinstance(payload, dict):
        text = payload.get("generated_text") or payload.get("text")
        if isinstance(text, str):
            return text.strip()
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            c0 = choices[0]
            if isinstance(c0, dict):
                ctext = c0.get("text")
                if isinstance(ctext, str):
                    return ctext.strip()
    return None


def _build_narrative_prompt(
    *,
    entity_id: str,
    template_narrative: str,
    rel_ctx: Dict[str, Any],
    regression: float,
    probability: float,
    ranking_score: float,
) -> str:
    return (
        "Rewrite the following deterministic model narrative into plain-text long form. "
        "Do not invent facts, entities, or scores. Keep numeric values consistent with the facts. "
        "Keep it to 5-8 sentences and plain language.\n\n"
        f"Entity ID: {entity_id}\n"
        f"Facts: {rel_ctx}\n"
        f"Scores: regression={regression:.4f}, probability={probability:.4f}, ranking={ranking_score:.4f}\n"
        f"Template: {template_narrative}\n"
    )


async def maybe_generate_long_narrative(
    *,
    entity_id: str,
    template_narrative: str,
    rel_ctx: Dict[str, Any],
    regression: float,
    probability: float,
    ranking_score: float,
) -> Tuple[str, bool, str]:
    settings = get_settings()
    if not settings.llm_enabled:
        return template_narrative, False, "llm_disabled"
    if not settings.llm_endpoint_url or not settings.llm_api_token:
        return template_narrative, False, "llm_not_configured"

    prompt = _build_narrative_prompt(
        entity_id=entity_id,
        template_narrative=template_narrative,
        rel_ctx=rel_ctx,
        regression=regression,
        probability=probability,
        ranking_score=ranking_score,
    )

    headers = {
        "Authorization": f"Bearer {settings.llm_api_token}",
        "Content-Type": "application/json",
    }
    request_payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": settings.llm_max_tokens,
            "temperature": settings.llm_temperature,
            "return_full_text": False,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
            response = await client.post(settings.llm_endpoint_url, headers=headers, json=request_payload)
            response.raise_for_status()
            payload = response.json()
            text = _extract_generated_text(payload)
            if not text:
                emit_performance_event(
                    "llm.narrative",
                    status="error",
                    provider=settings.llm_provider,
                    model=settings.llm_model_id,
                    reason="empty_response",
                )
                return template_narrative, False, "empty_response"
            emit_performance_event(
                "llm.narrative",
                status="ok",
                provider=settings.llm_provider,
                model=settings.llm_model_id,
                chars=len(text),
            )
            return text, True, "ok"
    except Exception as exc:  # noqa: BLE001
        emit_performance_event(
            "llm.narrative",
            status="error",
            provider=settings.llm_provider,
            model=settings.llm_model_id,
            reason=type(exc).__name__,
        )
        return template_narrative, False, type(exc).__name__


async def maybe_interpret_query(query: str) -> Tuple[str, bool]:
    settings = get_settings()
    if not settings.llm_enabled or not settings.llm_endpoint_url or not settings.llm_api_token:
        return "Keyword retrieval over entity IDs and deterministic predictions.", False

    headers = {
        "Authorization": f"Bearer {settings.llm_api_token}",
        "Content-Type": "application/json",
    }
    prompt = (
        "Summarize this analyst query as a single short interpretation for retrieval. "
        "No extra prose.\n"
        f"Query: {query}"
    )
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 80,
            "temperature": 0.1,
            "return_full_text": False,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
            response = await client.post(settings.llm_endpoint_url, headers=headers, json=payload)
            response.raise_for_status()
            text = _extract_generated_text(response.json())
            if text:
                emit_performance_event("llm.query", status="ok", model=settings.llm_model_id, chars=len(text))
                return text, True
    except Exception as exc:  # noqa: BLE001
        emit_performance_event("llm.query", status="error", reason=type(exc).__name__)
    return "Keyword retrieval over entity IDs and deterministic predictions.", False
