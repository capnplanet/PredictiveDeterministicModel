from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app


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
