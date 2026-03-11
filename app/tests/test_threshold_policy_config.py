from __future__ import annotations

import pytest

from app.training.train import _threshold_policies


@pytest.mark.unit
def test_threshold_policy_config_loads_from_json() -> None:
    policies = _threshold_policies()

    assert "v1" in policies
    assert "default" in policies["v1"]
    assert "reg_r2" in policies["v1"]["default"]
