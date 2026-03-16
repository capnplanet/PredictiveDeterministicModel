from __future__ import annotations

import threading
import uuid
from datetime import datetime
from typing import Optional, Tuple

from sqlalchemy import select

from app.core.config import get_settings
from app.core.performance import correlation_scope, emit_performance_event
from app.db.models import FeatureExtractionTask
from app.db.session import session_scope
from app.services.feature_extraction import extract_features_for_pending


def _execute_feature_extraction_task(task_id: str) -> None:
    started_at = datetime.utcnow()
    payload = {}

    with session_scope() as session:
        task = session.get(FeatureExtractionTask, task_id)
        if task is None:
            return
        task.status = "running"  # type: ignore[assignment]
        task.started_at = started_at
        task.error_message = None
        payload = dict(task.request_payload or {})

    correlation_id = (str(payload.get("correlation_id") or "").strip() or None)

    with correlation_scope(correlation_id):
        try:
            with session_scope() as session:
                updated = extract_features_for_pending(session)

            completed_at = datetime.utcnow()
            with session_scope() as session:
                task = session.get(FeatureExtractionTask, task_id)
                if task is None:
                    return
                task.status = "success"  # type: ignore[assignment]
                task.updated_artifacts = int(updated)
                task.completed_at = completed_at
                task.error_message = None

            emit_performance_event(
                "features.async.success",
                task_id=task_id,
                updated_artifacts=int(updated),
                duration_ms=(completed_at - started_at).total_seconds() * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001
            completed_at = datetime.utcnow()
            with session_scope() as session:
                task = session.get(FeatureExtractionTask, task_id)
                if task is None:
                    return
                task.status = "failed"  # type: ignore[assignment]
                task.completed_at = completed_at
                task.error_message = f"{type(exc).__name__}: {exc}"

            emit_performance_event(
                "features.async.failed",
                task_id=task_id,
                error_type=type(exc).__name__,
                duration_ms=(completed_at - started_at).total_seconds() * 1000.0,
            )


def _dispatch_feature_extraction_task(task_id: str) -> Optional[str]:
    settings = get_settings()
    if settings.celery_enabled:
        try:
            from celery import Celery

            app = Celery(
                "predictive_deterministic_model",
                broker=settings.celery_broker_url,
                backend=settings.celery_result_backend,
            )
            app.conf.task_default_queue = settings.queue_feature_extraction_name
            app.conf.task_always_eager = settings.celery_task_always_eager

            async_result = app.send_task(
                "app.worker.execute_feature_extraction_task",
                args=[task_id],
                queue=settings.queue_feature_extraction_name,
            )
            return str(async_result.id)
        except Exception as exc:  # noqa: BLE001
            if not settings.async_fallback_sync_execution:
                raise RuntimeError(f"Celery dispatch failed: {type(exc).__name__}: {exc}") from exc

    thread = threading.Thread(target=_execute_feature_extraction_task, args=(task_id,), daemon=True)
    thread.start()
    return None


def enqueue_feature_extraction_task(
    idempotency_key: Optional[str],
    correlation_id: Optional[str] = None,
) -> Tuple[str, bool]:
    normalized_key = (idempotency_key or "").strip() or None

    with session_scope() as session:
        if normalized_key is not None:
            existing = (
                session.execute(
                    select(FeatureExtractionTask)
                    .where(FeatureExtractionTask.idempotency_key == normalized_key)
                    .order_by(FeatureExtractionTask.created_at.desc(), FeatureExtractionTask.task_id.desc())
                )
                .scalars()
                .first()
            )
            if existing is not None:
                return existing.task_id, True

        settings = get_settings()
        task_id = uuid.uuid4().hex
        task = FeatureExtractionTask(
            task_id=task_id,
            status="pending",  # type: ignore[assignment]
            queue_name=settings.queue_feature_extraction_name,
            idempotency_key=normalized_key,
            request_payload={"correlation_id": (correlation_id or "").strip() or None},
        )
        session.add(task)

    celery_task_id: Optional[str] = None
    dispatch_error: Optional[str] = None
    try:
        celery_task_id = _dispatch_feature_extraction_task(task_id)
    except Exception as exc:  # noqa: BLE001
        dispatch_error = f"{type(exc).__name__}: {exc}"

    if celery_task_id is not None or dispatch_error is not None:
        with session_scope() as session:
            task = session.get(FeatureExtractionTask, task_id)
            if task is not None:
                task.celery_task_id = celery_task_id
                if dispatch_error is not None:
                    task.status = "failed"  # type: ignore[assignment]
                    task.completed_at = datetime.utcnow()
                    task.error_message = dispatch_error

    emit_performance_event(
        "features.async.enqueued",
        task_id=task_id,
        queue_name=get_settings().queue_feature_extraction_name,
        idempotency_key=normalized_key,
        correlation_id=(correlation_id or "").strip() or None,
        idempotent_reused=False,
        celery_enabled=get_settings().celery_enabled,
    )
    return task_id, False
