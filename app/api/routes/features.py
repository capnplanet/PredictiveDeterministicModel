from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import FeatureExtractionEnqueueRequest, FeatureExtractionTaskResponse
from app.core.performance import get_correlation_id
from app.core.security import principal_from_request, principal_to_context
from app.db.models import FeatureExtractionTask
from app.db.session import session_scope
from app.services.feature_extraction import extract_features_for_pending
from app.services.feature_tasks import enqueue_feature_extraction_task

router = APIRouter(prefix="/features", tags=["features"])


@router.post("/extract", response_model=dict)
async def extract_pending_features() -> dict:
    with session_scope() as session:
        count = extract_features_for_pending(session)
    return {"updated_artifacts": count}


@router.post("/extract/async", response_model=FeatureExtractionTaskResponse)
async def enqueue_extract_pending_features(
    request: FeatureExtractionEnqueueRequest,
    http_request: Request,
) -> FeatureExtractionTaskResponse:
    principal_context = principal_to_context(principal_from_request(http_request))
    task_id, _ = enqueue_feature_extraction_task(
        idempotency_key=request.idempotency_key,
        correlation_id=get_correlation_id(),
        principal_context=principal_context,
    )

    with session_scope() as session:
        task = session.get(FeatureExtractionTask, task_id)
        if task is None:
            raise RuntimeError(f"Feature extraction task not found after enqueue: {task_id}")
        return FeatureExtractionTaskResponse(
            task_id=task.task_id,
            status=str(task.status),
            queue_name=task.queue_name,
            idempotency_key=task.idempotency_key,
            correlation_id=(task.request_payload or {}).get("correlation_id"),
            updated_artifacts=task.updated_artifacts,
            error_message=task.error_message,
            created_at=task.created_at,
            started_at=task.started_at,
            completed_at=task.completed_at,
        )


@router.get("/extract/async/{task_id}", response_model=FeatureExtractionTaskResponse)
async def get_extract_pending_features_task(task_id: str) -> FeatureExtractionTaskResponse:
    with session_scope() as session:
        task = session.get(FeatureExtractionTask, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Feature extraction task not found: {task_id}")
        return FeatureExtractionTaskResponse(
            task_id=task.task_id,
            status=str(task.status),
            queue_name=task.queue_name,
            idempotency_key=task.idempotency_key,
            correlation_id=(task.request_payload or {}).get("correlation_id"),
            updated_artifacts=task.updated_artifacts,
            error_message=task.error_message,
            created_at=task.created_at,
            started_at=task.started_at,
            completed_at=task.completed_at,
        )
