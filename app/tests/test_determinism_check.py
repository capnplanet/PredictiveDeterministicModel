from __future__ import annotations

import pytest

from app.training.train import run_determinism_check


@pytest.mark.integration
@pytest.mark.determinism
def test_run_determinism_check() -> None:
    report = run_determinism_check()
    assert report["same_run_id"] is True
    assert report["same_metrics"] is True
    assert report["same_model_sha"] is True
    assert report["same_predictions"] is True
