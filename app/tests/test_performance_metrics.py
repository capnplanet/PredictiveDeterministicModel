from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.performance import performance_metrics_path
from app.main import app
from app.services.performance_report import build_performance_report


def test_api_request_performance_event_is_emitted() -> None:
    client = TestClient(app)
    response = client.get("/health/")
    assert response.status_code == 200

    metrics_path = performance_metrics_path()
    assert metrics_path.exists()

    events = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    assert any(event.get("event") == "api.request" for event in events)
    assert any(event.get("path") == "/health/" for event in events if event.get("event") == "api.request")


def test_build_performance_report_from_jsonl(tmp_path: Path) -> None:
    metrics_path = tmp_path / "performance_metrics.jsonl"
    metrics_path.write_text(
        "\n".join(
            [
                json.dumps({"event": "api.request", "status": "ok", "duration_ms": 10.0}),
                json.dumps({"event": "api.request", "status": "ok", "duration_ms": 20.0}),
                json.dumps({"event": "training.total", "status": "ok", "duration_ms": 100.0}),
            ]
        )
    )

    report = build_performance_report(metrics_path=metrics_path)

    assert report["total_events"] == 3
    assert report["event_counts"]["api.request"] == 2
    assert report["status_counts"]["ok"] == 3
    assert report["event_duration_ms"]["api.request"]["p95_ms"] == 20.0
