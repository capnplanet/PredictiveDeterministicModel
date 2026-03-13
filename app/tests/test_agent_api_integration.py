from __future__ import annotations

import asyncio
import os
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.db.models import AgentAuditEvent, AgentRun, AgentStep, ModelRun
from app.db.session import session_scope
from app.main import app
from app.services.agent_tools import AGENT_TOOL_REGISTRY, AgentToolSpec


@pytest.mark.integration
@pytest.mark.api
def test_create_agent_run_requires_feature_flag() -> None:
    get_settings.cache_clear()
    os.environ["AGENT_ENABLED"] = "false"

    client = TestClient(app)
    response = client.post(
        "/agents/runs",
        json={
            "goal": "Investigate strongest entities",
            "context": {"query": "strongest entities"},
        },
    )
    assert response.status_code == 403


@pytest.mark.integration
@pytest.mark.api
def test_agent_run_lifecycle_approval_and_status() -> None:
    get_settings.cache_clear()
    os.environ["AGENT_ENABLED"] = "true"
    os.environ["AGENT_REQUIRE_APPROVAL"] = "true"

    client = TestClient(app)

    create_response = client.post(
        "/agents/runs",
        json={
            "goal": "Search likely risk entities",
            "context": {"query": "likely risk entities", "limit": 3},
        },
    )
    assert create_response.status_code == 200
    created = create_response.json()
    run_id = created["agent_run_id"]

    assert created["status"] == "awaiting_approval"
    assert isinstance(created["plan"], list)
    assert len(created["plan"]) >= 1

    plan_response = client.get(f"/agents/runs/{run_id}/plan")
    assert plan_response.status_code == 200
    plan_payload = plan_response.json()
    assert plan_payload["agent_run_id"] == run_id
    assert plan_payload["status"] == "awaiting_approval"

    approve_response = client.post(
        f"/agents/runs/{run_id}/approve",
        json={"approved": True, "reviewer_notes": "approved for execution"},
    )
    assert approve_response.status_code == 200
    approved = approve_response.json()
    assert approved["status"] == "pending"

    status_response = client.get(f"/agents/runs/{run_id}")
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["status"] == "pending"
    assert status_payload["total_steps"] >= 1

    pause_response = client.post(
        f"/agents/runs/{run_id}/control",
        json={"action": "pause", "reason": "operator pause"},
    )
    assert pause_response.status_code == 200
    assert pause_response.json()["status"] == "paused"

    resume_response = client.post(
        f"/agents/runs/{run_id}/control",
        json={"action": "resume"},
    )
    assert resume_response.status_code == 200
    assert resume_response.json()["status"] == "pending"

    list_response = client.get("/agents/runs")
    assert list_response.status_code == 200
    runs = list_response.json()
    assert isinstance(runs, list)
    assert any(item["agent_run_id"] == run_id for item in runs)


@pytest.mark.integration
@pytest.mark.api
def test_agent_step_execute_failure_is_recorded() -> None:
    get_settings.cache_clear()
    os.environ["AGENT_ENABLED"] = "true"
    os.environ["AGENT_REQUIRE_APPROVAL"] = "false"

    client = TestClient(app)

    create_response = client.post(
        "/agents/runs",
        json={
            "goal": "Query results for a run that does not exist",
            "context": {"run_id": "missing_run_for_agent_api"},
        },
    )
    assert create_response.status_code == 200
    run_id = create_response.json()["agent_run_id"]

    execute_response = client.post(
        f"/agents/runs/{run_id}/steps/0/execute",
        json={"force_continue": False},
    )
    assert execute_response.status_code == 200
    step = execute_response.json()
    assert step["status"] == "failed"
    assert isinstance(step["error_message"], str)
    assert step["error_message"]

    status_response = client.get(f"/agents/runs/{run_id}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "failed"


@pytest.mark.integration
@pytest.mark.api
def test_agent_step_execute_rejects_unsupported_tool() -> None:
    get_settings.cache_clear()
    os.environ["AGENT_ENABLED"] = "true"
    os.environ["AGENT_REQUIRE_APPROVAL"] = "false"

    client = TestClient(app)

    create_response = client.post(
        "/agents/runs",
        json={
            "goal": "Search likely risk entities",
            "context": {"query": "likely risk entities"},
        },
    )
    assert create_response.status_code == 200
    run_id = create_response.json()["agent_run_id"]

    with session_scope() as session:
        step = session.query(AgentStep).filter(AgentStep.agent_run_id == run_id, AgentStep.step_index == 0).first()
        assert step is not None
        step.tool_name = "nonexistent_tool"

    execute_response = client.post(
        f"/agents/runs/{run_id}/steps/0/execute",
        json={"force_continue": False},
    )
    assert execute_response.status_code == 200
    step_payload = execute_response.json()
    assert step_payload["status"] == "failed"
    assert "Unsupported tool" in (step_payload["error_message"] or "")


@pytest.mark.integration
@pytest.mark.api
def test_agent_step_execute_timeout_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    os.environ["AGENT_ENABLED"] = "true"
    os.environ["AGENT_REQUIRE_APPROVAL"] = "false"
    os.environ["AGENT_STEP_TIMEOUT_SECONDS"] = "0.01"

    async def _slow_query_tool(arguments: dict[str, object]) -> dict[str, object]:
        await asyncio.sleep(0.2)
        return {"ok": True}

    original_spec = AGENT_TOOL_REGISTRY["query_entities"]
    monkeypatch.setitem(
        AGENT_TOOL_REGISTRY,
        "query_entities",
        AgentToolSpec(
            name=original_spec.name,
            description=original_spec.description,
            deterministic_safe=original_spec.deterministic_safe,
            idempotent=original_spec.idempotent,
            executor=_slow_query_tool,
        ),
    )

    client = TestClient(app)
    create_response = client.post(
        "/agents/runs",
        json={
            "goal": "Search likely risk entities",
            "context": {"query": "likely risk entities"},
        },
    )
    assert create_response.status_code == 200
    run_id = create_response.json()["agent_run_id"]

    execute_response = client.post(
        f"/agents/runs/{run_id}/steps/0/execute",
        json={"force_continue": False},
    )
    assert execute_response.status_code == 200
    step_payload = execute_response.json()
    assert step_payload["status"] == "failed"
    assert step_payload["error_message"] == "Step execution timed out"


@pytest.mark.integration
@pytest.mark.api
def test_agent_run_audit_events_are_recorded() -> None:
    get_settings.cache_clear()
    os.environ["AGENT_ENABLED"] = "true"
    os.environ["AGENT_REQUIRE_APPROVAL"] = "true"

    client = TestClient(app)
    create_response = client.post(
        "/agents/runs",
        json={
            "goal": "Search likely risk entities",
            "context": {"query": "likely risk entities"},
        },
    )
    assert create_response.status_code == 200
    run_id = create_response.json()["agent_run_id"]

    approve_response = client.post(
        f"/agents/runs/{run_id}/approve",
        json={"approved": True, "reviewer_notes": "approved for execution"},
    )
    assert approve_response.status_code == 200

    audit_response = client.get(f"/agents/runs/{run_id}/audit")
    assert audit_response.status_code == 200
    payload = audit_response.json()
    event_types = [event["event_type"] for event in payload]
    assert "run_created" in event_types
    assert "plan_reviewed" in event_types


@pytest.mark.integration
@pytest.mark.api
def test_agent_plan_includes_determinism_step_when_requested() -> None:
    get_settings.cache_clear()
    os.environ["AGENT_ENABLED"] = "true"
    os.environ["AGENT_REQUIRE_APPROVAL"] = "false"

    client = TestClient(app)
    create_response = client.post(
        "/agents/runs",
        json={
            "goal": "Run determinism verification for this run",
            "context": {"run_id": "placeholder_run_id"},
        },
    )
    assert create_response.status_code == 200
    plan = create_response.json()["plan"]
    tool_names = [step["tool_name"] for step in plan]
    assert "verify_run_determinism" in tool_names


@pytest.mark.integration
@pytest.mark.api
def test_agent_train_step_sets_model_run_lineage(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    os.environ["AGENT_ENABLED"] = "true"
    os.environ["AGENT_REQUIRE_APPROVAL"] = "false"
    os.environ["AGENT_ENFORCE_DETERMINISM"] = "false"

    seeded_run_id = "run_agent_lineage_seed"
    with session_scope() as session:
        session.merge(
            ModelRun(
                run_id=seeded_run_id,
                config={"seed": 1234},
                metrics={"loss": 0.0},
                model_sha256="seed_sha_agent_lineage",
                data_manifest={"source": "test"},
                status="success",  # type: ignore[assignment]
                logs_path="artifacts/test/log.jsonl",
                created_at=datetime.utcnow(),
            )
        )

    def _fake_run_training(config_path=None):  # type: ignore[no-untyped-def]
        return seeded_run_id, {"ok": 1.0}

    monkeypatch.setattr("app.services.agent_tools.run_training", _fake_run_training)

    client = TestClient(app)
    create_response = client.post(
        "/agents/runs",
        json={
            "goal": "Train a refreshed model",
            "context": {},
        },
    )
    assert create_response.status_code == 200
    run_id = create_response.json()["agent_run_id"]
    plan = create_response.json()["plan"]
    assert plan[0]["tool_name"] == "train_model"

    execute_response = client.post(
        f"/agents/runs/{run_id}/steps/0/execute",
        json={"force_continue": False},
    )
    assert execute_response.status_code == 200
    assert execute_response.json()["status"] == "success"

    with session_scope() as session:
        run = session.get(ModelRun, seeded_run_id)
        assert run is not None
        assert run.created_by_agent_run_id == run_id


@pytest.mark.integration
@pytest.mark.api
def test_agent_audit_events_are_immutable() -> None:
    get_settings.cache_clear()
    os.environ["AGENT_ENABLED"] = "true"
    os.environ["AGENT_REQUIRE_APPROVAL"] = "false"

    client = TestClient(app)
    create_response = client.post(
        "/agents/runs",
        json={
            "goal": "Search likely risk entities",
            "context": {"query": "likely risk entities"},
        },
    )
    assert create_response.status_code == 200
    run_id = create_response.json()["agent_run_id"]

    with pytest.raises(ValueError):
        with session_scope() as session:
            event = (
                session.query(AgentAuditEvent)
                .filter(AgentAuditEvent.agent_run_id == run_id)
                .order_by(AgentAuditEvent.created_at.asc())
                .first()
            )
            assert event is not None
            event.event_type = "tampered"
            session.flush()


@pytest.mark.integration
@pytest.mark.api
def test_agent_train_step_records_affected_run_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    os.environ["AGENT_ENABLED"] = "true"
    os.environ["AGENT_REQUIRE_APPROVAL"] = "false"
    os.environ["AGENT_ENFORCE_DETERMINISM"] = "false"

    seeded_run_id = "run_agent_metrics_seed"

    def _fake_run_training(config_path=None):  # type: ignore[no-untyped-def]
        return seeded_run_id, {"ok": 1.0}

    monkeypatch.setattr("app.services.agent_tools.run_training", _fake_run_training)

    client = TestClient(app)
    create_response = client.post(
        "/agents/runs",
        json={"goal": "Train model now", "context": {}},
    )
    assert create_response.status_code == 200
    run_id = create_response.json()["agent_run_id"]

    execute_response = client.post(
        f"/agents/runs/{run_id}/steps/0/execute",
        json={"force_continue": False},
    )
    assert execute_response.status_code == 200
    assert execute_response.json()["status"] == "success"

    status_response = client.get(f"/agents/runs/{run_id}")
    assert status_response.status_code == 200
    with session_scope() as session:
        run = session.get(ModelRun, seeded_run_id)
        # Run may not exist when training is mocked, but metric tracking must still be present on AgentRun.
        _ = run
        agent_run = session.get(AgentRun, run_id)
        assert agent_run is not None
        metrics = agent_run.metrics if isinstance(agent_run.metrics, dict) else {}
        assert seeded_run_id in list(metrics.get("affected_run_ids", []))


@pytest.mark.integration
@pytest.mark.api
def test_agent_train_step_fails_when_determinism_verification_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    os.environ["AGENT_ENABLED"] = "true"
    os.environ["AGENT_REQUIRE_APPROVAL"] = "false"
    os.environ["AGENT_ENFORCE_DETERMINISM"] = "true"

    async def _fake_train_tool(arguments: dict[str, object]) -> dict[str, object]:
        return {"run_id": "run_det_fail_seed", "metrics": {"ok": 1.0}}

    async def _failing_det_tool(arguments: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("Determinism verification failed for run run_det_fail_seed")

    train_spec = AGENT_TOOL_REGISTRY["train_model"]
    det_spec = AGENT_TOOL_REGISTRY["verify_run_determinism"]
    monkeypatch.setitem(
        AGENT_TOOL_REGISTRY,
        "train_model",
        AgentToolSpec(
            name=train_spec.name,
            description=train_spec.description,
            deterministic_safe=train_spec.deterministic_safe,
            idempotent=train_spec.idempotent,
            executor=_fake_train_tool,
        ),
    )
    monkeypatch.setitem(
        AGENT_TOOL_REGISTRY,
        "verify_run_determinism",
        AgentToolSpec(
            name=det_spec.name,
            description=det_spec.description,
            deterministic_safe=det_spec.deterministic_safe,
            idempotent=det_spec.idempotent,
            executor=_failing_det_tool,
        ),
    )

    client = TestClient(app)
    create_response = client.post(
        "/agents/runs",
        json={"goal": "Train refreshed model", "context": {}},
    )
    assert create_response.status_code == 200
    run_id = create_response.json()["agent_run_id"]

    execute_response = client.post(
        f"/agents/runs/{run_id}/steps/0/execute",
        json={"force_continue": False},
    )
    assert execute_response.status_code == 200
    payload = execute_response.json()
    assert payload["status"] == "failed"
    assert "Determinism verification failed" in (payload["error_message"] or "")
