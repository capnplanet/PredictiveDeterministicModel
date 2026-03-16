from __future__ import annotations

from app.core.config import get_settings
from app.services.batch_inference_tasks import _execute_batch_inference_task
from app.services.feature_tasks import _execute_feature_extraction_task
from app.services.training_tasks import _execute_training_task

try:
    from celery import Celery
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("Celery is required to run worker mode") from exc

settings = get_settings()
celery_app = Celery(
    "predictive_deterministic_model",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery_app.conf.task_default_queue = settings.queue_training_name
celery_app.conf.task_always_eager = settings.celery_task_always_eager


@celery_app.task(name="app.worker.execute_training_task")
def execute_training_task(task_id: str) -> None:
    _execute_training_task(task_id)


@celery_app.task(name="app.worker.execute_feature_extraction_task")
def execute_feature_extraction_task(task_id: str) -> None:
    _execute_feature_extraction_task(task_id)


@celery_app.task(name="app.worker.execute_batch_inference_task")
def execute_batch_inference_task(task_id: str) -> None:
    _execute_batch_inference_task(task_id)
