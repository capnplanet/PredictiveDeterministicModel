from __future__ import annotations

from typing import List

from fastapi import APIRouter
from sqlalchemy import select

from app.api.schemas import RunInfo, TrainRequest, TrainResponse
from app.core.performance import timed_performance_event
from app.db.models import ModelRun
from app.db.session import session_scope
from app.training.train import TrainConfig, run_training

router = APIRouter(prefix="", tags=["training"])


@router.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest) -> TrainResponse:
    cfg = request.config
    config_path = None
    if cfg is not None:
        from pathlib import Path
        import json

        tmp = Path("data/api_train_config.json")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(cfg.model_dump()))
        config_path = tmp

    with timed_performance_event(
        "api.train",
        has_config=cfg is not None,
        epochs=getattr(cfg, "epochs", None),
        batch_size=getattr(cfg, "batch_size", None),
    ):
        run_id, metrics = run_training(config_path=config_path)
    return TrainResponse(run_id=run_id, metrics=metrics)


@router.get("/runs", response_model=List[RunInfo])
async def list_runs() -> List[RunInfo]:
    with session_scope() as session:
        runs = session.execute(select(ModelRun)).scalars().all()
        return [
            RunInfo(
                run_id=r.run_id,
                created_at=r.created_at,
                config=r.config,
                metrics=r.metrics,
                model_sha256=r.model_sha256,
                data_manifest=r.data_manifest,
                status=str(r.status),
                logs_path=r.logs_path,
            )
            for r in runs
        ]


@router.get("/runs/{run_id}", response_model=RunInfo)
async def get_run(run_id: str) -> RunInfo:
    with session_scope() as session:
        run = session.get(ModelRun, run_id)
        if run is None:
            raise RuntimeError(f"Run not found: {run_id}")
        return RunInfo(
            run_id=run.run_id,
            created_at=run.created_at,
            config=run.config,
            metrics=run.metrics,
            model_sha256=run.model_sha256,
            data_manifest=run.data_manifest,
            status=str(run.status),
            logs_path=run.logs_path,
        )
