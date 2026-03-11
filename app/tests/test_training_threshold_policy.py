from __future__ import annotations

import pytest

from app.training.train import TrainConfig, _evaluate_thresholds


@pytest.mark.unit
def test_threshold_policy_passes_for_synthetic_baseline_metrics() -> None:
    cfg = TrainConfig(corpus_name="synthetic_baseline", threshold_policy_version="v1", enforce_thresholds=True)
    metrics = {
        "reg_r2": 0.90,
        "cls_f1": 0.95,
        "rank_ndcg@10": 0.92,
    }

    result = _evaluate_thresholds(metrics, cfg)
    assert result["passed"] is True
    assert result["violations"] == []


@pytest.mark.unit
def test_threshold_policy_flags_violations() -> None:
    cfg = TrainConfig(corpus_name="uci_adult", threshold_policy_version="v1", enforce_thresholds=True)
    metrics = {
        "cls_f1": 0.50,
        "cls_precision": 0.55,
        "cls_recall": 0.40,
    }

    result = _evaluate_thresholds(metrics, cfg)
    assert result["passed"] is False
    assert len(result["violations"]) >= 1


@pytest.mark.unit
def test_threshold_policy_unknown_version_raises() -> None:
    cfg = TrainConfig(threshold_policy_version="does-not-exist")
    with pytest.raises(ValueError):
        _evaluate_thresholds({"reg_r2": 1.0}, cfg)
