from __future__ import annotations

import numpy as np
import pytest

from app.training.train import _split_entity_indices


@pytest.mark.unit
def test_split_entity_indices_is_deterministic_and_disjoint() -> None:
    train1, val1, test1 = _split_entity_indices(
        n_entities=64,
        seed=1234,
        val_fraction=0.2,
        test_fraction=0.2,
    )
    train2, val2, test2 = _split_entity_indices(
        n_entities=64,
        seed=1234,
        val_fraction=0.2,
        test_fraction=0.2,
    )

    assert np.array_equal(train1, train2)
    assert np.array_equal(val1, val2)
    assert np.array_equal(test1, test2)

    assert train1.size > 0
    assert val1.size > 0
    assert test1.size > 0

    all_indices = np.concatenate([train1, val1, test1])
    assert np.unique(all_indices).size == 64
    assert set(train1).isdisjoint(set(val1))
    assert set(train1).isdisjoint(set(test1))
    assert set(val1).isdisjoint(set(test1))


@pytest.mark.unit
def test_split_entity_indices_tiny_dataset_falls_back_to_train_only() -> None:
    train_idx, val_idx, test_idx = _split_entity_indices(
        n_entities=4,
        seed=42,
        val_fraction=0.2,
        test_fraction=0.2,
    )

    assert train_idx.size == 4
    assert val_idx.size == 0
    assert test_idx.size == 0


@pytest.mark.unit
def test_split_entity_indices_rejects_invalid_fractions() -> None:
    with pytest.raises(ValueError):
        _split_entity_indices(n_entities=10, seed=1, val_fraction=-0.1, test_fraction=0.2)

    with pytest.raises(ValueError):
        _split_entity_indices(n_entities=10, seed=1, val_fraction=0.7, test_fraction=0.4)


@pytest.mark.unit
def test_split_entity_indices_time_strategy_uses_created_order() -> None:
    created = np.array([30.0, 10.0, 40.0, 20.0, 50.0, 60.0], dtype="float64")
    train_idx, val_idx, test_idx = _split_entity_indices(
        n_entities=6,
        seed=999,
        val_fraction=0.2,
        test_fraction=0.2,
        split_strategy="time",
        created_at_unix=created,
    )

    # Sorted created_at order gives indices [1,3,0,2,4,5].
    # With fractions above, we expect 4 train / 1 val / 1 test.
    assert np.array_equal(train_idx, np.array([0, 1, 2, 3], dtype="int64"))
    assert np.array_equal(val_idx, np.array([4], dtype="int64"))
    assert np.array_equal(test_idx, np.array([5], dtype="int64"))
