from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict

from app.api.routes.predict import predict as predict_entities
from app.api.routes.query import query_predictions
from app.api.schemas import PredictRequest, QueryRequest
from app.db.models import ModelRun
from app.db.session import session_scope
from app.training.train import reproduce_run

ToolExecutor = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


@dataclass(frozen=True)
class AgentToolSpec:
    name: str
    description: str
    deterministic_safe: bool
    idempotent: bool
    executor: ToolExecutor


async def _tool_get_run_metrics(arguments: Dict[str, Any]) -> Dict[str, Any]:
    run_id = str(arguments.get("run_id", "")).strip()
    if not run_id:
        raise RuntimeError("run_id is required")

    with session_scope() as session:
        run = session.get(ModelRun, run_id)
        if run is None:
            raise RuntimeError(f"Run not found: {run_id}")
        return {
            "run_id": run.run_id,
            "status": str(run.status),
            "metrics": run.metrics,
            "created_at": run.created_at.isoformat(),
            "model_sha256": run.model_sha256,
        }


async def _tool_predict_entities(arguments: Dict[str, Any]) -> Dict[str, Any]:
    ids = arguments.get("entity_ids")
    if not isinstance(ids, list) or not ids:
        raise RuntimeError("entity_ids must be a non-empty list")

    response = await predict_entities(
        PredictRequest(
            entity_ids=[str(v) for v in ids],
            run_id=arguments.get("run_id"),
            explanations=False,
            narrative_mode=arguments.get("narrative_mode", "template"),
        )
    )
    return {
        "run_id": response.run_id,
        "prediction_count": len(response.predictions),
        "predictions": [item.model_dump() for item in response.predictions],
    }


async def _tool_query_entities(arguments: Dict[str, Any]) -> Dict[str, Any]:
    query = str(arguments.get("query", "")).strip()
    if not query:
        raise RuntimeError("query is required")

    response = await query_predictions(
        QueryRequest(
            query=query,
            run_id=arguments.get("run_id"),
            limit=int(arguments.get("limit", 5)),
        )
    )
    return response.model_dump()


async def _tool_verify_run_determinism(arguments: Dict[str, Any]) -> Dict[str, Any]:
    run_id = str(arguments.get("run_id", "")).strip()
    if not run_id:
        raise RuntimeError("run_id is required")

    report = reproduce_run(run_id)
    checks = ["same_run_id", "same_model_sha", "same_metrics", "same_predictions"]
    failing_checks = [key for key in checks if not bool(report.get(key))]
    if failing_checks:
        failing = ", ".join(failing_checks)
        raise RuntimeError(f"Determinism verification failed for run {run_id}: {failing}")

    return {
        "run_id": run_id,
        "determinism_verified": True,
        "report": report,
    }


AGENT_TOOL_REGISTRY: Dict[str, AgentToolSpec] = {
    "get_run_metrics": AgentToolSpec(
        name="get_run_metrics",
        description="Retrieve persisted metadata and metrics for a model run.",
        deterministic_safe=True,
        idempotent=True,
        executor=_tool_get_run_metrics,
    ),
    "predict_entities": AgentToolSpec(
        name="predict_entities",
        description="Run deterministic inference for one or more entities.",
        deterministic_safe=True,
        idempotent=True,
        executor=_tool_predict_entities,
    ),
    "query_entities": AgentToolSpec(
        name="query_entities",
        description="Search likely entities and return ranked query results.",
        deterministic_safe=True,
        idempotent=True,
        executor=_tool_query_entities,
    ),
    "verify_run_determinism": AgentToolSpec(
        name="verify_run_determinism",
        description="Reproduce a run and verify deterministic parity of outputs.",
        deterministic_safe=True,
        idempotent=True,
        executor=_tool_verify_run_determinism,
    ),
}


async def dispatch_agent_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    spec = AGENT_TOOL_REGISTRY.get(tool_name)
    if spec is None:
        available = ", ".join(sorted(AGENT_TOOL_REGISTRY.keys()))
        raise RuntimeError(f"Unsupported tool: {tool_name}. Available tools: {available}")
    return await spec.executor(arguments)
