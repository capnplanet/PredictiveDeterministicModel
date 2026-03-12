from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

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
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            c0 = choices[0]
            if isinstance(c0, dict):
                msg = c0.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content.strip()
        text = payload.get("generated_text") or payload.get("text")
        if isinstance(text, str):
            return text.strip()
        if isinstance(choices, list) and choices:
            c0 = choices[0]
            if isinstance(c0, dict):
                ctext = c0.get("text")
                if isinstance(ctext, str):
                    return ctext.strip()
    return None


def _chat_completions_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1/chat/completions"):
        return base_url
    return base_url.rstrip("/") + "/v1/chat/completions"


def _looks_like_hf_endpoint(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return "endpoints.huggingface.cloud" in host


async def _post_to_llm(settings, *, prompt: str) -> Tuple[Optional[str], str]:
    headers = {
        "Authorization": f"Bearer {settings.llm_api_token}",
        "Content-Type": "application/json",
    }

    # Hugging Face dedicated endpoints are commonly OpenAI-compatible at /v1/chat/completions.
    if _looks_like_hf_endpoint(settings.llm_endpoint_url):
        chat_url = _chat_completions_url(settings.llm_endpoint_url)
        chat_payload = {
            "model": settings.llm_model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": settings.llm_max_tokens,
            "temperature": settings.llm_temperature,
        }
        try:
            async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
                response = await client.post(chat_url, headers=headers, json=chat_payload)
                response.raise_for_status()
                return _extract_generated_text(response.json()), "chat"
        except Exception:  # noqa: BLE001
            pass

    legacy_payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": settings.llm_max_tokens,
            "temperature": settings.llm_temperature,
            "return_full_text": False,
        },
    }
    async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
        response = await client.post(settings.llm_endpoint_url, headers=headers, json=legacy_payload)
        response.raise_for_status()
        return _extract_generated_text(response.json()), "legacy"


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

    try:
        text, mode = await _post_to_llm(settings, prompt=prompt)
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
            mode=mode,
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

    prompt = (
        "Summarize this analyst query as a single short interpretation for retrieval. "
        "No extra prose.\n"
        f"Query: {query}"
    )
    try:
        text, mode = await _post_to_llm(settings, prompt=prompt)
        if text:
            emit_performance_event("llm.query", status="ok", model=settings.llm_model_id, mode=mode, chars=len(text))
            return text, True
    except Exception as exc:  # noqa: BLE001
        emit_performance_event("llm.query", status="error", reason=type(exc).__name__)
    return "Keyword retrieval over entity IDs and deterministic predictions.", False
