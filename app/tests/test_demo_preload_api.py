from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.mark.integration
@pytest.mark.api
def test_demo_preload_endpoint_populates_all_ingestion_points() -> None:
    client = TestClient(app)

    response = client.post(
        "/demo/preload",
        json={
            "profile": "small",
            "reset_existing": True,
            "extract_features": True,
            "train_model": False,
        },
    )
    assert response.status_code == 200
    body = response.json()

    assert body["profile"] == "small"
    assert body["entities"]["success_rows"] > 0
    assert body["events"]["success_rows"] > 0
    assert body["interactions"]["success_rows"] > 0
    assert body["artifacts_manifest"]["success_rows"] > 0
    assert body["single_artifact"]["artifact_type"] == "image"
    assert isinstance(body["features"]["updated_artifacts"], int)
    assert body["training"] is None
