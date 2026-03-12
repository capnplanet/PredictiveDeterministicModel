from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.api.routes.predict import predict as predict_entities
from app.api.routes.query import query_predictions
from app.api.schemas import (
    AgentControlRequest,
    AgentGoalRequest,
    AgentPlanApprovalRequest,
    AgentPlanResponse,
    AgentRunResponse,
    AgentStatusResponse,
    AgentStepExecuteRequest,
    AgentStepPlan,
    AgentStepResult,
    PredictRequest,
    QueryRequest,
)
from app.core.config import get_settings
from app.db.models import AgentRun, AgentStep, ModelRun
from app.db.session import session_scope

router = APIRouter(prefix="/agents", tags=["agent"])


def _coerce_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _step_to_plan(step: AgentStep) -> AgentStepPlan:
    return AgentStepPlan(
        step_index=step.step_index,
        tool_name=step.tool_name,
        arguments=_coerce_dict(step.arguments),
        status=str(step.status),
        retry_count=step.retry_count,
        output=_coerce_dict(step.output) if step.output is not None else None,
        error_message=step.error_message,
    )


def _run_to_response(run: AgentRun, steps: List[AgentStep]) -> AgentRunResponse:
    return AgentRunResponse(
        agent_run_id=run.agent_run_id,
        created_at=run.created_at,
        updated_at=run.updated_at,
        goal=run.goal,
        status=str(run.status),
        current_step_index=run.current_step_index,
        max_steps=run.max_steps,
        step_retries=run.step_retries,
        require_approval=bool(run.require_approval),
        context=_coerce_dict(run.run_context),
        metrics=_coerce_dict(run.metrics),
        plan=[_step_to_plan(step) for step in steps],
        last_error=run.last_error,
    )


def _plan_steps(goal: str, context: Dict[str, Any], max_steps: int) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    lowered = goal.lower()

    run_id = context.get("run_id")
    entity_ids = context.get("entity_ids")
    query_text = context.get("query")

    if isinstance(run_id, str) and run_id.strip():
        steps.append(
            {
                "tool_name": "get_run_metrics",
                "arguments": {"run_id": run_id.strip()},
            }
        )

    if isinstance(entity_ids, list) and entity_ids:
        ids = [str(v) for v in entity_ids if isinstance(v, (str, int, float))]
        if ids:
            steps.append(
                {
                    "tool_name": "predict_entities",
                    "arguments": {
                        "entity_ids": ids,
                        "run_id": run_id if isinstance(run_id, str) and run_id.strip() else None,
                        "narrative_mode": "template",
                    },
                }
            )

    if "query" in lowered or "search" in lowered:
        query_value = query_text if isinstance(query_text, str) and query_text.strip() else goal
        steps.append(
            {
                "tool_name": "query_entities",
                "arguments": {
                    "query": query_value,
                    "run_id": run_id if isinstance(run_id, str) and run_id.strip() else None,
                    "limit": 5,
                },
            }
        )

    if not steps:
        steps.append(
            {
                "tool_name": "query_entities",
                "arguments": {
                    "query": goal,
                    "run_id": run_id if isinstance(run_id, str) and run_id.strip() else None,
                    "limit": 5,
                },
            }
        )

    return steps[: max_steps]


async def _invoke_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name == "get_run_metrics":
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
            }

    if tool_name == "predict_entities":
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

    if tool_name == "query_entities":
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

    raise RuntimeError(f"Unsupported tool: {tool_name}")


def _next_pending_step(steps: List[AgentStep]) -> AgentStep | None:
    for step in steps:
        if str(step.status) == "pending":
            return step
    return None


async def _execute_single_step(run: AgentRun, step: AgentStep, force_continue: bool) -> Tuple[AgentStep, bool]:
    now = datetime.utcnow()
    step.status = "running"
    step.started_at = now
    run.status = "executing"
    run.updated_at = now

    try:
        result = await _invoke_tool(step.tool_name, _coerce_dict(step.arguments))
        step.status = "success"
        step.output = result
        step.error_message = None
        step.completed_at = datetime.utcnow()
        run.current_step_index = max(run.current_step_index, step.step_index + 1)
        run.updated_at = datetime.utcnow()
        return step, True
    except Exception as exc:  # noqa: BLE001
        step.retry_count += 1
        step.error_message = str(exc)
        step.completed_at = datetime.utcnow()

        if step.retry_count <= run.step_retries and force_continue:
            step.status = "pending"
        else:
            step.status = "failed"
            run.status = "failed"
            run.last_error = str(exc)
        run.updated_at = datetime.utcnow()
        return step, False


@router.post("/runs", response_model=AgentRunResponse)
async def create_agent_run(request: AgentGoalRequest) -> AgentRunResponse:
    settings = get_settings()
    if not settings.agent_enabled:
        raise HTTPException(status_code=403, detail="Agent runtime is disabled")

    run_id = f"agr_{uuid4().hex[:24]}"
    max_steps = max(1, settings.agent_max_plan_steps)
    plan_payload = _plan_steps(request.goal, request.context, max_steps=max_steps)
    require_approval = bool(settings.agent_require_approval)
    initial_status = "awaiting_approval" if require_approval else "pending"

    with session_scope() as session:
        run = AgentRun(
            agent_run_id=run_id,
            goal=request.goal,
            status=initial_status,
            plan=plan_payload,
            current_step_index=0,
            max_steps=max_steps,
            step_retries=max(0, settings.agent_max_step_retries),
            require_approval=require_approval,
            run_context=request.context,
            metrics={"steps_total": len(plan_payload), "steps_succeeded": 0, "steps_failed": 0},
        )
        session.add(run)
        session.flush()

        for idx, entry in enumerate(plan_payload):
            session.add(
                AgentStep(
                    agent_run_id=run.agent_run_id,
                    step_index=idx,
                    tool_name=str(entry.get("tool_name", "")),
                    arguments=_coerce_dict(entry.get("arguments")),
                    status="pending",
                )
            )

        session.flush()
        steps = list(session.execute(select(AgentStep).where(AgentStep.agent_run_id == run.agent_run_id).order_by(AgentStep.step_index)).scalars().all())
        return _run_to_response(run, steps)


@router.get("/runs/{agent_run_id}/plan", response_model=AgentPlanResponse)
async def get_agent_plan(agent_run_id: str) -> AgentPlanResponse:
    with session_scope() as session:
        run = session.get(AgentRun, agent_run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Agent run not found")
        steps = list(
            session.execute(
                select(AgentStep).where(AgentStep.agent_run_id == agent_run_id).order_by(AgentStep.step_index)
            ).scalars().all()
        )
        return AgentPlanResponse(
            agent_run_id=run.agent_run_id,
            status=str(run.status),
            goal=run.goal,
            plan=[_step_to_plan(step) for step in steps],
        )


@router.post("/runs/{agent_run_id}/approve", response_model=AgentRunResponse)
async def approve_agent_plan(agent_run_id: str, request: AgentPlanApprovalRequest) -> AgentRunResponse:
    with session_scope() as session:
        run = session.get(AgentRun, agent_run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Agent run not found")
        steps = list(
            session.execute(
                select(AgentStep).where(AgentStep.agent_run_id == agent_run_id).order_by(AgentStep.step_index)
            ).scalars().all()
        )

        if not bool(run.require_approval):
            return _run_to_response(run, steps)

        if str(run.status) != "awaiting_approval":
            raise HTTPException(status_code=409, detail=f"Run is not awaiting approval: {run.status}")

        if request.approved:
            run.status = "pending"
            notes = request.reviewer_notes.strip() if request.reviewer_notes else "approved"
            run.metrics = {**_coerce_dict(run.metrics), "review_note": notes}
        else:
            run.status = "aborted"
            notes = request.reviewer_notes.strip() if request.reviewer_notes else "rejected"
            run.metrics = {**_coerce_dict(run.metrics), "review_note": notes}
        run.updated_at = datetime.utcnow()

        session.flush()
        return _run_to_response(run, steps)


@router.post("/runs/{agent_run_id}/steps/{step_index}/execute", response_model=AgentStepResult)
async def execute_agent_step(agent_run_id: str, step_index: int, request: AgentStepExecuteRequest) -> AgentStepResult:
    with session_scope() as session:
        run = session.get(AgentRun, agent_run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Agent run not found")

        if str(run.status) in {"awaiting_approval", "paused", "aborted", "completed"}:
            raise HTTPException(status_code=409, detail=f"Run state does not allow execution: {run.status}")

        step = session.execute(
            select(AgentStep).where(AgentStep.agent_run_id == agent_run_id, AgentStep.step_index == step_index)
        ).scalar_one_or_none()
        if step is None:
            raise HTTPException(status_code=404, detail="Agent step not found")

        step, ok = await _execute_single_step(run, step, force_continue=request.force_continue)

        steps = list(
            session.execute(
                select(AgentStep).where(AgentStep.agent_run_id == agent_run_id).order_by(AgentStep.step_index)
            ).scalars().all()
        )

        succeeded = sum(1 for item in steps if str(item.status) == "success")
        failed = sum(1 for item in steps if str(item.status) == "failed")
        pending = sum(1 for item in steps if str(item.status) == "pending")
        run.metrics = {
            **_coerce_dict(run.metrics),
            "steps_total": len(steps),
            "steps_succeeded": succeeded,
            "steps_failed": failed,
        }

        if ok and pending == 0 and failed == 0:
            run.status = "completed"
            run.updated_at = datetime.utcnow()

        session.flush()
        return AgentStepResult(
            agent_run_id=run.agent_run_id,
            step_id=str(step.step_id),
            step_index=step.step_index,
            tool_name=step.tool_name,
            status=str(step.status),
            retry_count=step.retry_count,
            output=_coerce_dict(step.output) if step.output is not None else None,
            error_message=step.error_message,
        )


@router.post("/runs/{agent_run_id}/loop", response_model=AgentRunResponse)
async def execute_agent_loop(agent_run_id: str) -> AgentRunResponse:
    with session_scope() as session:
        run = session.get(AgentRun, agent_run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Agent run not found")

        if str(run.status) == "awaiting_approval":
            raise HTTPException(status_code=409, detail="Plan approval is required before loop execution")
        if str(run.status) in {"paused", "aborted", "completed"}:
            raise HTTPException(status_code=409, detail=f"Run state does not allow loop execution: {run.status}")

        steps = list(
            session.execute(
                select(AgentStep).where(AgentStep.agent_run_id == agent_run_id).order_by(AgentStep.step_index)
            ).scalars().all()
        )

        iterations = 0
        while iterations < run.max_steps:
            step = _next_pending_step(steps)
            if step is None:
                break
            _, ok = await _execute_single_step(run, step, force_continue=True)
            iterations += 1
            if not ok and str(run.status) == "failed":
                break

        succeeded = sum(1 for item in steps if str(item.status) == "success")
        failed = sum(1 for item in steps if str(item.status) == "failed")
        pending = sum(1 for item in steps if str(item.status) == "pending")

        run.metrics = {
            **_coerce_dict(run.metrics),
            "steps_total": len(steps),
            "steps_succeeded": succeeded,
            "steps_failed": failed,
            "loop_iterations": iterations,
        }

        if failed == 0 and pending == 0:
            run.status = "completed"
        run.updated_at = datetime.utcnow()

        session.flush()
        return _run_to_response(run, steps)


@router.post("/runs/{agent_run_id}/control", response_model=AgentRunResponse)
async def control_agent_run(agent_run_id: str, request: AgentControlRequest) -> AgentRunResponse:
    with session_scope() as session:
        run = session.get(AgentRun, agent_run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Agent run not found")

        if request.action == "pause":
            if str(run.status) not in {"pending", "executing"}:
                raise HTTPException(status_code=409, detail=f"Run cannot be paused from state: {run.status}")
            run.status = "paused"
        elif request.action == "resume":
            if str(run.status) != "paused":
                raise HTTPException(status_code=409, detail=f"Run cannot be resumed from state: {run.status}")
            run.status = "pending"
        elif request.action == "abort":
            if str(run.status) in {"completed", "aborted"}:
                raise HTTPException(status_code=409, detail=f"Run cannot be aborted from state: {run.status}")
            run.status = "aborted"
        run.updated_at = datetime.utcnow()

        metrics = _coerce_dict(run.metrics)
        if request.reason:
            metrics["control_reason"] = request.reason
        run.metrics = metrics

        steps = list(
            session.execute(
                select(AgentStep).where(AgentStep.agent_run_id == agent_run_id).order_by(AgentStep.step_index)
            ).scalars().all()
        )
        session.flush()
        return _run_to_response(run, steps)


@router.get("/runs/{agent_run_id}", response_model=AgentStatusResponse)
async def get_agent_run_status(agent_run_id: str) -> AgentStatusResponse:
    with session_scope() as session:
        run = session.get(AgentRun, agent_run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Agent run not found")
        steps = list(
            session.execute(
                select(AgentStep).where(AgentStep.agent_run_id == agent_run_id).order_by(AgentStep.step_index)
            ).scalars().all()
        )
        completed_steps = sum(1 for step in steps if str(step.status) == "success")
        failed_steps = sum(1 for step in steps if str(step.status) == "failed")
        return AgentStatusResponse(
            agent_run_id=run.agent_run_id,
            status=str(run.status),
            current_step_index=run.current_step_index,
            total_steps=len(steps),
            completed_steps=completed_steps,
            failed_steps=failed_steps,
            goal=run.goal,
            updated_at=run.updated_at,
            last_error=run.last_error,
        )


@router.get("/runs", response_model=List[AgentStatusResponse])
async def list_agent_runs() -> List[AgentStatusResponse]:
    with session_scope() as session:
        runs = list(session.execute(select(AgentRun).order_by(AgentRun.created_at.desc())).scalars().all())
        responses: List[AgentStatusResponse] = []
        for run in runs:
            steps = list(
                session.execute(
                    select(AgentStep).where(AgentStep.agent_run_id == run.agent_run_id).order_by(AgentStep.step_index)
                ).scalars().all()
            )
            responses.append(
                AgentStatusResponse(
                    agent_run_id=run.agent_run_id,
                    status=str(run.status),
                    current_step_index=run.current_step_index,
                    total_steps=len(steps),
                    completed_steps=sum(1 for step in steps if str(step.status) == "success"),
                    failed_steps=sum(1 for step in steps if str(step.status) == "failed"),
                    goal=run.goal,
                    updated_at=run.updated_at,
                    last_error=run.last_error,
                )
            )
        return responses
