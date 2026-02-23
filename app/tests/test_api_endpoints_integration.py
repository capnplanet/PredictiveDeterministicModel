from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.mark.integration
@pytest.mark.api
def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


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
    assert prediction.get("explanation") is not None
