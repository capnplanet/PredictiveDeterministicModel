from __future__ import annotations

from app.training.train import run_determinism_check


def test_run_determinism_check() -> None:
    report = run_determinism_check()
    assert report["same_run_id"] is True
    assert report["same_metrics"] is True
    assert report["same_model_sha"] is True
    assert report["same_predictions"] is True
