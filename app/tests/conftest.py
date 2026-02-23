from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def deterministic_test_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(1234)
    np.random.seed(1234)

    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("NUMEXPR_NUM_THREADS", "1")

    monkeypatch.setenv("ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
