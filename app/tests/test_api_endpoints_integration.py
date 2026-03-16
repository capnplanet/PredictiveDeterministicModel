from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.db.models import BatchInferenceTask, Entity, FeatureExtractionTask, TrainingTask
from app.db.session import session_scope
from app.main import app


@pytest.mark.integration
@pytest.mark.api
def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health/")

    assert response.status_code in (200, 503)
    if response.status_code == 200:
        assert response.json() == {"status": "ok", "database": "ok"}
    else:
        assert response.json() == {"detail": {"status": "degraded", "database": "error"}}


@pytest.mark.integration
@pytest.mark.api
def test_ingest_train_predict_api_flow() -> None:
    client = TestClient(app)
    suffix = uuid.uuid4().hex[:8]
    entity_a = f"E_API_{suffix}_A"
    entity_b = f"E_API_{suffix}_B"

    entities_csv = (
        "entity_id,attributes,created_at\n"
        f'{entity_a},"{{""x"":0.1,""y"":0.2,""z"":0.3,""target_regression"":0.7,""target_binary"":1,""target_ranking"":0.9}}",2025-01-01T00:00:00\n'
        f'{entity_b},"{{""x"":0.2,""y"":0.1,""z"":0.4,""target_regression"":0.4,""target_binary"":0,""target_ranking"":0.2}}",2025-01-01T00:01:00\n'
    )
    ingest_entities = client.post(
        "/ingest/entities",
        files={"file": ("entities.csv", entities_csv, "text/csv")},
    )
    assert ingest_entities.status_code == 200
    body = ingest_entities.json()
    assert body["total_rows"] == 2
    assert body["success_rows"] == 2
    assert body["failed_rows"] == 0

    events_csv = (
        "timestamp,entity_id,event_type,event_value,event_metadata\n"
        f'2025-01-01T00:10:00,{entity_a},purchase,10.5,"{{""source"":""api-test""}}"\n'
        f'2025-01-01T00:11:00,{entity_b},purchase,2.0,"{{""source"":""api-test""}}"\n'
    )
    ingest_events = client.post(
        "/ingest/events",
        files={"file": ("events.csv", events_csv, "text/csv")},
    )
    assert ingest_events.status_code == 200
    assert ingest_events.json()["success_rows"] == 2

    interactions_csv = (
        "timestamp,src_entity_id,dst_entity_id,interaction_type,interaction_value,metadata\n"
        f'2025-01-01T00:12:00,{entity_a},{entity_b},linked,1,"{{""source"":""api-test""}}"\n'
    )
    ingest_interactions = client.post(
        "/ingest/interactions",
        files={"file": ("interactions.csv", interactions_csv, "text/csv")},
    )
    assert ingest_interactions.status_code == 200
    assert ingest_interactions.json()["success_rows"] == 1

    extract_features = client.post("/features/extract")
    assert extract_features.status_code == 200
    assert isinstance(extract_features.json().get("updated_artifacts"), int)

    train_response = client.post(
        "/train",
        json={
            "config": {
                "epochs": 1,
                "batch_size": 2,
                "lr": 0.001,
                "seed": 1234,
            }
        },
    )
    assert train_response.status_code == 200
    train_body = train_response.json()
    assert isinstance(train_body["run_id"], str)
    assert train_body["run_id"]
    assert "metrics" in train_body

    runs_response = client.get("/runs")
    assert runs_response.status_code == 200
    runs = runs_response.json()
    assert isinstance(runs, list)
    assert any(run["run_id"] == train_body["run_id"] for run in runs)

    run_detail = client.get(f"/runs/{train_body['run_id']}")
    assert run_detail.status_code == 200
    assert run_detail.json()["run_id"] == train_body["run_id"]

    predict_response = client.post(
        "/predict",
        json={
            "entity_ids": [entity_a],
            "run_id": train_body["run_id"],
            "explanations": True,
        },
    )
    assert predict_response.status_code == 200
    pred_body = predict_response.json()
    assert pred_body["run_id"] == train_body["run_id"]
    assert len(pred_body["predictions"]) == 1
    prediction = pred_body["predictions"][0]
    assert prediction["entity_id"] == entity_a
    assert "regression" in prediction
    assert "probability" in prediction
    assert "ranking_score" in prediction
    assert isinstance(prediction.get("embedding"), list)
    assert "narrative" in prediction
    assert isinstance(prediction["narrative"], str)
    assert prediction["narrative"].strip()
    assert prediction.get("explanation") is not None

    query_response = client.post(
        "/query",
        json={
            "query": entity_a,
            "run_id": train_body["run_id"],
            "limit": 3,
        },
    )
    assert query_response.status_code == 200
    query_body = query_response.json()
    assert query_body["run_id"] == train_body["run_id"]
    assert query_body["query"] == entity_a
    assert isinstance(query_body["interpreted_as"], str)
    assert isinstance(query_body["llm_used"], bool)
    assert isinstance(query_body["results"], list)
    assert any(item["entity_id"] == entity_a for item in query_body["results"])

    strongest_query = client.post(
        "/query",
        json={
            "query": "show entities with strongest relationship signals",
            "run_id": train_body["run_id"],
            "limit": 2,
        },
    )
    assert strongest_query.status_code == 200
    strongest_results = strongest_query.json()["results"]
    assert len(strongest_results) >= 1
    strongest_scores = [float(item["ranking_score"]) for item in strongest_results]
    assert strongest_scores == sorted(strongest_scores, reverse=True)

    relationship_query = client.post(
        "/query",
        json={
            "query": "show relationship patterns across entities",
            "run_id": train_body["run_id"],
            "limit": 2,
        },
    )
    assert relationship_query.status_code == 200
    relationship_results = relationship_query.json()["results"]
    assert len(relationship_results) >= 1
    relationship_scores = [float(item["ranking_score"]) for item in relationship_results]
    assert relationship_scores == sorted(relationship_scores, reverse=True)

    weakest_query = client.post(
        "/query",
        json={
            "query": "show entities with weakest relationship signals",
            "run_id": train_body["run_id"],
            "limit": 2,
        },
    )
    assert weakest_query.status_code == 200
    weakest_results = weakest_query.json()["results"]
    assert len(weakest_results) >= 1
    weakest_scores = [float(item["ranking_score"]) for item in weakest_results]
    assert weakest_scores == sorted(weakest_scores)


@pytest.mark.integration
@pytest.mark.api
def test_async_train_enqueue_and_status_idempotency(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep this test deterministic and fast by preventing background execution.
    monkeypatch.setattr("app.services.training_tasks._dispatch_training_task", lambda _task_id: None)

    client = TestClient(app)

    enqueue_payload = {
        "idempotency_key": "train-async-idem-001",
        "config": {
            "epochs": 1,
            "batch_size": 2,
            "lr": 0.001,
            "seed": 1234,
        },
    }
    first = client.post("/train/async", json=enqueue_payload)
    assert first.status_code == 200
    first_body = first.json()
    assert isinstance(first_body["task_id"], str)
    assert first_body["task_id"]
    assert first_body["status"] == "pending"
    assert first_body["idempotency_key"] == "train-async-idem-001"

    second = client.post("/train/async", json=enqueue_payload)
    assert second.status_code == 200
    second_body = second.json()
    assert second_body["task_id"] == first_body["task_id"]
    assert second_body["idempotency_key"] == "train-async-idem-001"

    status = client.get(f"/train/async/{first_body['task_id']}")
    assert status.status_code == 200
    status_body = status.json()
    assert status_body["task_id"] == first_body["task_id"]
    assert status_body["status"] in {"pending", "running", "success", "failed"}


@pytest.mark.integration
@pytest.mark.api
def test_async_feature_extract_enqueue_and_status_idempotency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.feature_tasks._dispatch_feature_extraction_task", lambda _task_id: None)

    client = TestClient(app)
    payload = {"idempotency_key": "extract-async-idem-001"}

    first = client.post("/features/extract/async", json=payload)
    assert first.status_code == 200
    first_body = first.json()
    assert isinstance(first_body["task_id"], str)
    assert first_body["status"] == "pending"
    assert first_body["idempotency_key"] == "extract-async-idem-001"

    second = client.post("/features/extract/async", json=payload)
    assert second.status_code == 200
    second_body = second.json()
    assert second_body["task_id"] == first_body["task_id"]

    status = client.get(f"/features/extract/async/{first_body['task_id']}")
    assert status.status_code == 200
    status_body = status.json()
    assert status_body["task_id"] == first_body["task_id"]
    assert status_body["status"] in {"pending", "running", "success", "failed"}


@pytest.mark.integration
@pytest.mark.api
def test_async_batch_predict_enqueue_and_status_idempotency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.batch_inference_tasks._dispatch_batch_inference_task", lambda _task_id: None)

    client = TestClient(app)
    suffix = uuid.uuid4().hex[:8]
    entity_id = f"E_ASYNC_{suffix}"
    payload = {
        "entity_ids": [entity_id],
        "idempotency_key": "predict-async-idem-001",
        "explanations": False,
        "narrative_mode": "template",
    }

    first = client.post("/predict/async", json=payload)
    assert first.status_code == 200
    first_body = first.json()
    assert isinstance(first_body["task_id"], str)
    assert first_body["status"] == "pending"
    assert first_body["idempotency_key"] == "predict-async-idem-001"

    second = client.post("/predict/async", json=payload)
    assert second.status_code == 200
    second_body = second.json()
    assert second_body["task_id"] == first_body["task_id"]

    status = client.get(f"/predict/async/{first_body['task_id']}")
    assert status.status_code == 200
    status_body = status.json()
    assert status_body["task_id"] == first_body["task_id"]
    assert status_body["status"] in {"pending", "running", "success", "failed"}


@pytest.mark.integration
@pytest.mark.api
def test_queue_health_endpoint_returns_queue_backlog_shape() -> None:
    client = TestClient(app)
    response = client.get("/health/queues")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ok", "degraded"}
    assert isinstance(payload.get("broker"), dict)
    assert isinstance(payload.get("queues"), dict)
    assert "training" in payload["queues"]
    assert "extraction" in payload["queues"]
    assert "batch_inference" in payload["queues"]
    for queue_name in ("training", "extraction", "batch_inference"):
        queue_payload = payload["queues"][queue_name]
        assert "backlog" in queue_payload
        assert "oldest_pending_age_seconds" in queue_payload
        assert "saturation_ratio" in queue_payload
        assert "max_concurrency" in queue_payload


@pytest.mark.integration
@pytest.mark.api
def test_async_task_endpoints_propagate_request_correlation_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.training_tasks._dispatch_training_task", lambda _task_id: None)
    monkeypatch.setattr("app.services.feature_tasks._dispatch_feature_extraction_task", lambda _task_id: None)
    monkeypatch.setattr("app.services.batch_inference_tasks._dispatch_batch_inference_task", lambda _task_id: None)

    client = TestClient(app)
    correlation_id = "corr-phase4-001"
    headers = {"x-correlation-id": correlation_id}

    train_response = client.post(
        "/train/async",
        headers=headers,
        json={"idempotency_key": "corr-train-001", "config": {"epochs": 1, "batch_size": 2}},
    )
    assert train_response.status_code == 200
    assert train_response.headers.get("x-correlation-id") == correlation_id
    train_task = train_response.json()
    assert train_task["correlation_id"] == correlation_id

    extract_response = client.post(
        "/features/extract/async",
        headers=headers,
        json={"idempotency_key": "corr-extract-001"},
    )
    assert extract_response.status_code == 200
    assert extract_response.headers.get("x-correlation-id") == correlation_id
    extract_task = extract_response.json()
    assert extract_task["correlation_id"] == correlation_id

    suffix = uuid.uuid4().hex[:8]
    predict_response = client.post(
        "/predict/async",
        headers=headers,
        json={
            "entity_ids": [f"E_CORR_{suffix}"],
            "idempotency_key": "corr-predict-001",
            "explanations": False,
            "narrative_mode": "template",
        },
    )
    assert predict_response.status_code == 200
    assert predict_response.headers.get("x-correlation-id") == correlation_id
    predict_task = predict_response.json()
    assert predict_task["correlation_id"] == correlation_id

    with session_scope() as session:
        stored_train = session.get(TrainingTask, train_task["task_id"])
        stored_extract = session.get(FeatureExtractionTask, extract_task["task_id"])
        stored_predict = session.get(BatchInferenceTask, predict_task["task_id"])

        assert stored_train is not None
        assert stored_extract is not None
        assert stored_predict is not None
        assert (stored_train.request_payload or {}).get("correlation_id") == correlation_id
        assert (stored_extract.request_payload or {}).get("correlation_id") == correlation_id
        assert (stored_predict.request_payload or {}).get("correlation_id") == correlation_id


@pytest.mark.integration
@pytest.mark.api
def test_async_completion_events_include_correlation_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from app.services import feature_tasks as feature_tasks_module

    def _dispatch_inline(task_id: str) -> None:
        feature_tasks_module._execute_feature_extraction_task(task_id)
        return None

    monkeypatch.setattr("app.services.feature_tasks._dispatch_feature_extraction_task", _dispatch_inline)

    client = TestClient(app)
    correlation_id = "corr-async-event-001"
    unique_idempotency = f"corr-async-event-{uuid.uuid4().hex[:12]}"

    response = client.post(
        "/features/extract/async",
        headers={"x-correlation-id": correlation_id},
        json={"idempotency_key": unique_idempotency},
    )
    assert response.status_code == 200

    metrics_path = tmp_path / "data" / "performance_metrics.jsonl"
    assert metrics_path.exists()

    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    success_rows = [row for row in rows if row.get("event") == "features.async.success"]
    assert success_rows
    assert any(row.get("correlation_id") == correlation_id for row in success_rows)


@pytest.mark.integration
@pytest.mark.api
def test_ingest_entities_api_checkpoint_resume_restart_behavior() -> None:
    client = TestClient(app)
    suffix = uuid.uuid4().hex[:8]
    entity_a = f"E_CPR_{suffix}_A"
    entity_b = f"E_CPR_{suffix}_B"
    checkpoint_key = f"entities-restart-{suffix}"

    entities_csv = (
        "entity_id,attributes,created_at\n"
        f'{entity_a},"{{""x"":0.1,""y"":0.2,""z"":0.3}}",2025-01-01T00:00:00\n'
        f'{entity_b},"{{""x"":0.2,""y"":0.3,""z"":0.4}}",2025-01-01T00:01:00\n'
    )

    first = client.post(
        "/ingest/entities",
        data={
            "chunk_size": "1",
            "checkpoint_key": checkpoint_key,
            "resume_from_checkpoint": "true",
        },
        files={"file": ("entities.csv", entities_csv, "text/csv")},
    )
    assert first.status_code == 200
    first_body = first.json()
    assert first_body["success_rows"] == 2

    second = client.post(
        "/ingest/entities",
        data={
            "chunk_size": "1",
            "checkpoint_key": checkpoint_key,
            "resume_from_checkpoint": "true",
        },
        files={"file": ("entities.csv", entities_csv, "text/csv")},
    )
    assert second.status_code == 200
    second_body = second.json()
    assert second_body["success_rows"] == 0

    with session_scope() as session:
        ids = {
            item.entity_id
            for item in session.query(Entity).filter(Entity.entity_id.in_([entity_a, entity_b])).all()
        }
        assert ids == {entity_a, entity_b}
