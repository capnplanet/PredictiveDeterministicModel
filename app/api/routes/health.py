from fastapi import APIRouter, HTTPException
from sqlalchemy import func, text

from app.core.config import get_settings
from app.core.performance import emit_performance_event
from app.db.models import BatchInferenceTask, FeatureExtractionTask, TrainingTask
from app.db.session import session_scope

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
async def health_check() -> dict[str, str]:
    try:
        with session_scope() as session:
            session.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail={"status": "degraded", "database": "error"}) from exc
    return {"status": "ok", "database": "ok"}


@router.get("/queues")
async def queue_health() -> dict:
    settings = get_settings()

    with session_scope() as session:
        session.execute(text("SELECT 1"))

        def _counts(model, max_concurrency: int) -> dict[str, int | float | None]:  # type: ignore[no-untyped-def]
            from datetime import datetime

            rows = session.query(model.status, func.count(model.task_id)).group_by(model.status).all()
            out = {"pending": 0, "running": 0, "success": 0, "failed": 0}
            for status, count in rows:
                out[str(status)] = int(count)
            out["backlog"] = int(out.get("pending", 0) + out.get("running", 0))
            oldest_pending = (
                session.query(func.min(model.created_at))
                .filter(model.status == "pending")
                .scalar()
            )
            if oldest_pending is None:
                out["oldest_pending_age_seconds"] = None
            else:
                age_seconds = (datetime.now(oldest_pending.tzinfo) - oldest_pending).total_seconds()
                out["oldest_pending_age_seconds"] = round(max(age_seconds, 0.0), 3)

            running = int(out.get("running", 0))
            safe_capacity = max(int(max_concurrency), 1)
            out["max_concurrency"] = safe_capacity
            out["saturation_ratio"] = round(running / float(safe_capacity), 3)
            return out

        training = _counts(TrainingTask, settings.queue_training_max_concurrency)
        extraction = _counts(FeatureExtractionTask, settings.queue_feature_extraction_max_concurrency)
        batch_inference = _counts(BatchInferenceTask, settings.queue_batch_inference_max_concurrency)

    broker_ok = False
    broker_error = None
    try:
        import redis

        client = redis.Redis.from_url(settings.celery_broker_url, socket_connect_timeout=1.0, socket_timeout=1.0)
        broker_ok = bool(client.ping())
    except Exception as exc:  # noqa: BLE001
        broker_error = f"{type(exc).__name__}: {exc}"

    payload = {
        "status": "ok" if broker_ok else "degraded",
        "broker": {
            "ok": broker_ok,
            "url": settings.celery_broker_url,
            "error": broker_error,
        },
        "queues": {
            settings.queue_training_name: training,
            settings.queue_feature_extraction_name: extraction,
            settings.queue_batch_inference_name: batch_inference,
        },
    }
    emit_performance_event(
        "health.queues",
        status="ok" if broker_ok else "error",
        broker_ok=broker_ok,
        training_backlog=training["backlog"],
        extraction_backlog=extraction["backlog"],
        batch_inference_backlog=batch_inference["backlog"],
        training_oldest_pending_age_seconds=training["oldest_pending_age_seconds"],
        extraction_oldest_pending_age_seconds=extraction["oldest_pending_age_seconds"],
        batch_inference_oldest_pending_age_seconds=batch_inference["oldest_pending_age_seconds"],
        training_saturation_ratio=training["saturation_ratio"],
        extraction_saturation_ratio=extraction["saturation_ratio"],
        batch_inference_saturation_ratio=batch_inference["saturation_ratio"],
    )
    return payload
