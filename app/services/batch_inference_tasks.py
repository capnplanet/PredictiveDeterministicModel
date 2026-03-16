from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from sqlalchemy import select

from app.core.config import get_settings
from app.core.performance import correlation_scope, emit_performance_event
from app.db.models import BatchInferenceTask
from app.db.session import session_scope


def _execute_batch_inference_task(task_id: str) -> None:
    started_at = datetime.utcnow()
    payload: Dict[str, Any] = {}

    with session_scope() as session:
        task = session.get(BatchInferenceTask, task_id)
        if task is None:
            return
        task.status = "running"  # type: ignore[assignment]
        task.started_at = started_at
        task.error_message = None
        payload = dict(task.request_payload or {})

    correlation_id = (str(payload.get("correlation_id") or "").strip() or None)

    with correlation_scope(correlation_id):
        try:
            from app.api.routes.predict import predict
            from app.api.schemas import PredictRequest

            response = asyncio.run(
                predict(
                    PredictRequest(
                        entity_ids=list(payload.get("entity_ids") or []),
                        run_id=payload.get("run_id"),
                        explanations=bool(payload.get("explanations", False)),
                        narrative_mode=str(payload.get("narrative_mode", "template")),
                    )
                )
            )

            completed_at = datetime.utcnow()
            with session_scope() as session:
                task = session.get(BatchInferenceTask, task_id)
                if task is None:
                    return
                task.status = "success"  # type: ignore[assignment]
                task.completed_at = completed_at
                task.run_id = response.run_id
                task.result_payload = response.model_dump()
                task.error_message = None

            emit_performance_event(
                "predict.async.success",
                task_id=task_id,
                run_id=response.run_id,
                duration_ms=(completed_at - started_at).total_seconds() * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001
            completed_at = datetime.utcnow()
            with session_scope() as session:
                task = session.get(BatchInferenceTask, task_id)
                if task is None:
                    return
                task.status = "failed"  # type: ignore[assignment]
                task.completed_at = completed_at
                task.error_message = f"{type(exc).__name__}: {exc}"

            emit_performance_event(
                "predict.async.failed",
                task_id=task_id,
                error_type=type(exc).__name__,
                duration_ms=(completed_at - started_at).total_seconds() * 1000.0,
            )


def _dispatch_batch_inference_task(task_id: str) -> Optional[str]:
    settings = get_settings()
    if settings.celery_enabled:
        try:
            from celery import Celery

            app = Celery(
                "predictive_deterministic_model",
                broker=settings.celery_broker_url,
                backend=settings.celery_result_backend,
            )
            app.conf.task_default_queue = settings.queue_batch_inference_name
            app.conf.task_always_eager = settings.celery_task_always_eager

            async_result = app.send_task(
                "app.worker.execute_batch_inference_task",
                args=[task_id],
                queue=settings.queue_batch_inference_name,
            )
            return str(async_result.id)
        except Exception as exc:  # noqa: BLE001
            if not settings.async_fallback_sync_execution:
                raise RuntimeError(f"Celery dispatch failed: {type(exc).__name__}: {exc}") from exc

    thread = threading.Thread(target=_execute_batch_inference_task, args=(task_id,), daemon=True)
    thread.start()
    return None


def _canonical_payload_key(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def enqueue_batch_inference_task(request_payload: Dict[str, Any], idempotency_key: Optional[str]) -> Tuple[str, bool]:
    payload = dict(request_payload)
    normalized_key = (idempotency_key or "").strip() or _canonical_payload_key(payload)

    with session_scope() as session:
        existing = (
            session.execute(
                select(BatchInferenceTask)
                .where(BatchInferenceTask.idempotency_key == normalized_key)
                .order_by(BatchInferenceTask.created_at.desc(), BatchInferenceTask.task_id.desc())
            )
            .scalars()
            .first()
        )
        if existing is not None:
            return existing.task_id, True

        settings = get_settings()
        task_id = uuid.uuid4().hex
        task = BatchInferenceTask(
            task_id=task_id,
            status="pending",  # type: ignore[assignment]
            queue_name=settings.queue_batch_inference_name,
            idempotency_key=normalized_key,
            request_payload=payload,
        )
        session.add(task)

    celery_task_id: Optional[str] = None
    dispatch_error: Optional[str] = None
    try:
        celery_task_id = _dispatch_batch_inference_task(task_id)
    except Exception as exc:  # noqa: BLE001
        dispatch_error = f"{type(exc).__name__}: {exc}"

    if celery_task_id is not None or dispatch_error is not None:
        with session_scope() as session:
            task = session.get(BatchInferenceTask, task_id)
            if task is not None:
                task.celery_task_id = celery_task_id
                if dispatch_error is not None:
                    task.status = "failed"  # type: ignore[assignment]
                    task.completed_at = datetime.utcnow()
                    task.error_message = dispatch_error

    emit_performance_event(
        "predict.async.enqueued",
        task_id=task_id,
        queue_name=get_settings().queue_batch_inference_name,
        idempotency_key=normalized_key,
        correlation_id=(payload.get("correlation_id") or None),
        idempotent_reused=False,
        celery_enabled=get_settings().celery_enabled,
    )
    return task_id, False
