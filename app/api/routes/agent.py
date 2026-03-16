from __future__ import annotations

import asyncio
from datetime import datetime
from time import perf_counter
from typing import Any, Dict, List, Tuple
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.api.schemas import (
    AgentApprovalSummaryResponse,
    AgentAuditEventResponse,
    AgentControlRequest,
    AgentDeterminismAuditResponse,
    AgentGoalRequest,
    AgentPlanApprovalRequest,
    AgentPlanResponse,
    AgentRunResponse,
    AgentStatusResponse,
    AgentStepExecuteRequest,
    AgentStepPlan,
    AgentStepResult,
)
from app.core.config import get_settings
from app.core.performance import emit_performance_event
from app.db.models import AgentAuditEvent, AgentRun, AgentStep
from app.db.session import session_scope
from app.services.agent_tools import dispatch_agent_tool

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
    queue_heavy = bool(context.get("queue_heavy", False))

    run_id = context.get("run_id")
    entity_ids = context.get("entity_ids")
    query_text = context.get("query")
    config_path = context.get("config_path")
    force_determinism = bool(context.get("force_determinism_check", False))

    if "train" in lowered or "retrain" in lowered:
        steps.append(
            {
                "tool_name": "enqueue_train_model" if queue_heavy else "train_model",
                "arguments": {
                    "config_path": str(config_path) if isinstance(config_path, str) and config_path.strip() else None,
                    "idempotency_key": str(context.get("idempotency_key", "")).strip() or None,
                },
            }
        )

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
                    "tool_name": "enqueue_batch_inference" if queue_heavy else "predict_entities",
                    "arguments": {
                        "entity_ids": ids,
                        "run_id": run_id if isinstance(run_id, str) and run_id.strip() else None,
                        "narrative_mode": "template",
                        "idempotency_key": str(context.get("idempotency_key", "")).strip() or None,
                    },
                }
            )

    if queue_heavy and bool(context.get("extract_features", False)):
        steps.append(
            {
                "tool_name": "enqueue_feature_extraction",
                "arguments": {
                    "idempotency_key": str(context.get("idempotency_key", "")).strip() or None,
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

    if (
        ("determinism" in lowered or "reproduc" in lowered or force_determinism)
        and isinstance(run_id, str)
        and run_id.strip()
    ):
        steps.append(
            {
                "tool_name": "verify_run_determinism",
                "arguments": {"run_id": run_id.strip()},
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


def _record_audit_event(
    *,
    session,
    run_id: str,
    event_type: str,
    actor: str = "system",
    details: Dict[str, Any] | None = None,
) -> None:
    session.add(
        AgentAuditEvent(
            agent_run_id=run_id,
            event_type=event_type,
            actor=actor,
            details=_coerce_dict(details),
        )
    )


async def _invoke_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return await dispatch_agent_tool(tool_name=tool_name, arguments=arguments)


def _next_pending_step(steps: List[AgentStep]) -> AgentStep | None:
    for step in steps:
        if str(step.status) == "pending":
            return step
    return None


def _append_unique_metric_value(run: AgentRun, key: str, value: str) -> None:
    metrics = _coerce_dict(run.metrics)
    current = metrics.get(key)
    values = [str(v) for v in current] if isinstance(current, list) else []
    if value not in values:
        values.append(value)
    metrics[key] = values
    run.metrics = metrics


def _append_unique_metric_values(run: AgentRun, key: str, values: List[str]) -> None:
    for value in values:
        trimmed = value.strip()
        if trimmed:
            _append_unique_metric_value(run, key, trimmed)


async def _execute_single_step(run: AgentRun, step: AgentStep, force_continue: bool) -> Tuple[AgentStep, bool]:
    started = perf_counter()
    now = datetime.utcnow()
    step.status = "running"
    step.started_at = now
    run.status = "executing"
    run.updated_at = now

    try:
        timeout_seconds = max(0.1, float(get_settings().agent_step_timeout_seconds))
        step_arguments = {
            **_coerce_dict(step.arguments),
            "agent_run_id": run.agent_run_id,
        }
        result = await asyncio.wait_for(
            _invoke_tool(step.tool_name, step_arguments),
            timeout=timeout_seconds,
        )

        if step.tool_name == "train_model":
            run_id_value = str(_coerce_dict(result).get("run_id", "")).strip()
            if run_id_value:
                _append_unique_metric_value(run, "affected_run_ids", run_id_value)
                if bool(get_settings().agent_enforce_determinism):
                    det_result = await asyncio.wait_for(
                        _invoke_tool(
                            "verify_run_determinism",
                            {
                                "run_id": run_id_value,
                                "agent_run_id": run.agent_run_id,
                            },
                        ),
                        timeout=timeout_seconds,
                    )
                    result = {
                        **_coerce_dict(result),
                        "determinism": _coerce_dict(det_result),
                    }

        if step.tool_name == "predict_entities":
            predictions = _coerce_dict(result).get("predictions")
            if isinstance(predictions, list):
                affected_entity_ids = [
                    str(item.get("entity_id", "")).strip()
                    for item in predictions
                    if isinstance(item, dict) and str(item.get("entity_id", "")).strip()
                ]
                _append_unique_metric_values(run, "affected_entity_ids", affected_entity_ids)

        if step.tool_name == "query_entities":
            query_results = _coerce_dict(result).get("results")
            if isinstance(query_results, list):
                affected_entity_ids = [
                    str(item.get("entity_id", "")).strip()
                    for item in query_results
                    if isinstance(item, dict) and str(item.get("entity_id", "")).strip()
                ]
                _append_unique_metric_values(run, "affected_entity_ids", affected_entity_ids)

        step.status = "success"
        step.output = result
        step.error_message = None
        step.completed_at = datetime.utcnow()
        run.current_step_index = max(run.current_step_index, step.step_index + 1)
        run.updated_at = datetime.utcnow()

        emit_performance_event(
            "agent.step.execute",
            status="ok",
            duration_ms=(perf_counter() - started) * 1000.0,
            agent_run_id=run.agent_run_id,
            step_index=step.step_index,
            tool_name=step.tool_name,
            retry_count=step.retry_count,
        )
        return step, True
    except asyncio.TimeoutError:
        step.retry_count += 1
        step.error_message = "Step execution timed out"
        step.completed_at = datetime.utcnow()

        if step.retry_count <= run.step_retries and force_continue:
            step.status = "pending"
        else:
            step.status = "failed"
            run.status = "failed"
            run.last_error = step.error_message
        run.updated_at = datetime.utcnow()

        emit_performance_event(
            "agent.step.execute",
            status="error",
            duration_ms=(perf_counter() - started) * 1000.0,
            agent_run_id=run.agent_run_id,
            step_index=step.step_index,
            tool_name=step.tool_name,
            retry_count=step.retry_count,
            error_type="TimeoutError",
            error=step.error_message,
        )
        return step, False
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

        emit_performance_event(
            "agent.step.execute",
            status="error",
            duration_ms=(perf_counter() - started) * 1000.0,
            agent_run_id=run.agent_run_id,
            step_index=step.step_index,
            tool_name=step.tool_name,
            retry_count=step.retry_count,
            error_type=type(exc).__name__,
            error=str(exc),
        )
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

        _record_audit_event(
            session=session,
            run_id=run.agent_run_id,
            event_type="run_created",
            details={
                "status": initial_status,
                "steps_total": len(plan_payload),
                "require_approval": require_approval,
            },
        )

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
        steps = list(
            session.execute(
                select(AgentStep).where(AgentStep.agent_run_id == run.agent_run_id).order_by(AgentStep.step_index)
            ).scalars().all()
        )
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

        _record_audit_event(
            session=session,
            run_id=run.agent_run_id,
            event_type="plan_reviewed",
            actor="reviewer",
            details={
                "approved": request.approved,
                "reviewer_notes": request.reviewer_notes or "",
                "new_status": str(run.status),
            },
        )

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

        _record_audit_event(
            session=session,
            run_id=run.agent_run_id,
            event_type="step_executed",
            details={
                "step_index": step.step_index,
                "tool_name": step.tool_name,
                "status": str(step.status),
                "retry_count": step.retry_count,
                "error_message": step.error_message,
            },
        )

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

        _record_audit_event(
            session=session,
            run_id=run.agent_run_id,
            event_type="loop_executed",
            details={
                "iterations": iterations,
                "steps_succeeded": succeeded,
                "steps_failed": failed,
                "steps_pending": pending,
                "status": str(run.status),
            },
        )

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
            previous_status = str(run.status)
            run.status = "paused"
        elif request.action == "resume":
            if str(run.status) != "paused":
                raise HTTPException(status_code=409, detail=f"Run cannot be resumed from state: {run.status}")
            previous_status = str(run.status)
            run.status = "pending"
        elif request.action == "abort":
            if str(run.status) in {"completed", "aborted"}:
                raise HTTPException(status_code=409, detail=f"Run cannot be aborted from state: {run.status}")
            previous_status = str(run.status)
            run.status = "aborted"
        run.updated_at = datetime.utcnow()

        metrics = _coerce_dict(run.metrics)
        if request.reason:
            metrics["control_reason"] = request.reason
        run.metrics = metrics

        _record_audit_event(
            session=session,
            run_id=run.agent_run_id,
            event_type="run_controlled",
            actor="operator",
            details={
                "action": request.action,
                "reason": request.reason or "",
                "previous_status": previous_status,
                "new_status": str(run.status),
            },
        )

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


@router.get("/runs/{agent_run_id}/audit", response_model=List[AgentAuditEventResponse])
async def list_agent_run_audit_events(agent_run_id: str) -> List[AgentAuditEventResponse]:
    with session_scope() as session:
        run = session.get(AgentRun, agent_run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Agent run not found")

        events = list(
            session.execute(
                select(AgentAuditEvent)
                .where(AgentAuditEvent.agent_run_id == agent_run_id)
                .order_by(AgentAuditEvent.created_at.asc())
            ).scalars().all()
        )

        return [
            AgentAuditEventResponse(
                event_id=str(event.event_id),
                agent_run_id=event.agent_run_id,
                created_at=event.created_at,
                event_type=event.event_type,
                actor=event.actor,
                details=_coerce_dict(event.details),
            )
            for event in events
        ]


@router.get("/compliance/approval-summary", response_model=AgentApprovalSummaryResponse)
async def get_agent_approval_summary() -> AgentApprovalSummaryResponse:
    with session_scope() as session:
        runs = list(session.execute(select(AgentRun)).scalars().all())

        total_runs = len(runs)
        approval_required_runs = sum(1 for run in runs if bool(run.require_approval))
        pending_approval_runs = sum(1 for run in runs if str(run.status) == "awaiting_approval")

        reviewed_events = list(
            session.execute(
                select(AgentAuditEvent).where(AgentAuditEvent.event_type == "plan_reviewed")
            ).scalars().all()
        )
        approved_runs = 0
        rejected_runs = 0
        for event in reviewed_events:
            details = _coerce_dict(event.details)
            if bool(details.get("approved", False)):
                approved_runs += 1
            else:
                rejected_runs += 1

        approval_rate = 0.0
        reviewed_total = approved_runs + rejected_runs
        if reviewed_total > 0:
            approval_rate = approved_runs / reviewed_total

        return AgentApprovalSummaryResponse(
            total_runs=total_runs,
            approval_required_runs=approval_required_runs,
            approved_runs=approved_runs,
            rejected_runs=rejected_runs,
            pending_approval_runs=pending_approval_runs,
            approval_rate=approval_rate,
        )


@router.get("/compliance/determinism-audit", response_model=AgentDeterminismAuditResponse)
async def get_agent_determinism_audit() -> AgentDeterminismAuditResponse:
    with session_scope() as session:
        runs = list(session.execute(select(AgentRun)).scalars().all())
        failing_run_ids = [
            run.agent_run_id
            for run in runs
            if isinstance(run.last_error, str) and "Determinism verification failed" in run.last_error
        ]
        return AgentDeterminismAuditResponse(
            total_runs=len(runs),
            deterministic_failures=len(failing_run_ids),
            failing_run_ids=failing_run_ids,
        )
