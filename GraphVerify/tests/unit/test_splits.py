"""Tests for dataset/splits.py."""
from __future__ import annotations

import pytest

from dataset.splits import apply_split, build_split, load_split, save_split


def test_build_split_partitions_all_ids_without_overlap():
    ids = [f"id{i}" for i in range(20)]
    split = build_split(ids, "demo", dev_fraction=0.3, seed=0)
    assert set(split.dev_ids) | set(split.test_ids) == set(ids)
    assert set(split.dev_ids) & set(split.test_ids) == set()
    assert len(split.dev_ids) == round(20 * 0.3)


def test_build_split_is_deterministic_given_seed():
    ids = [f"id{i}" for i in range(20)]
    a = build_split(ids, "demo", dev_fraction=0.3, seed=5)
    b = build_split(ids, "demo", dev_fraction=0.3, seed=5)
    assert a.dev_ids == b.dev_ids
    assert a.test_ids == b.test_ids


def test_build_split_different_seeds_differ():
    ids = [f"id{i}" for i in range(30)]
    a = build_split(ids, "demo", dev_fraction=0.5, seed=1)
    b = build_split(ids, "demo", dev_fraction=0.5, seed=2)
    assert a.dev_ids != b.dev_ids


def test_build_split_rejects_duplicates():
    with pytest.raises(ValueError):
        build_split(["a", "a", "b"], "demo")


def test_build_split_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        build_split(["a", "b"], "demo", dev_fraction=1.5)
    with pytest.raises(ValueError):
        build_split(["a", "b"], "demo", dev_fraction=0.0)


def test_save_and_load_split_roundtrip(tmp_path):
    ids = [f"id{i}" for i in range(10)]
    split = build_split(ids, "demo", dev_fraction=0.4, seed=3)
    path = str(tmp_path / "split.json")
    save_split(split, path)
    loaded = load_split(path)
    assert loaded.dev_ids == split.dev_ids
    assert loaded.test_ids == split.test_ids
    assert loaded.seed == split.seed


def test_apply_split_filters_records():
    ids = [f"id{i}" for i in range(10)]
    split = build_split(ids, "demo", dev_fraction=0.3, seed=0)
    records = [{"id": i} for i in ids]
    dev = apply_split(records, split, "dev")
    test = apply_split(records, split, "test")
    assert len(dev) == len(split.dev_ids)
    assert len(test) == len(split.test_ids)
    assert len(dev) + len(test) == len(records)


def test_apply_split_invalid_subset_raises():
    split = build_split(["a", "b"], "demo")
    with pytest.raises(ValueError):
        apply_split([{"id": "a"}], split, "not_a_real_subset")
