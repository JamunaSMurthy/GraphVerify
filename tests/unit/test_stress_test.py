"""Tests for dataset/stress_test.py."""
from __future__ import annotations

import pytest

from dataset.stress_test import (
    corrupt_numeric_and_date_mentions,
    inject_distractor_passages,
    inject_entity_alias_noise,
    perturb_top_k,
    remove_bridge_evidence,
)

PASSAGES = [
    {"id": "p1", "text": "Albert Einstein was born in Ulm in 1879.", "rank": 1, "score": 0.9, "title": "Einstein"},
    {"id": "p2", "text": "The Nobel Prize was awarded with 500000 SEK.", "rank": 2, "score": 0.8, "title": "Nobel"},
]


def test_perturb_top_k_truncates_and_renumbers():
    result = perturb_top_k(PASSAGES, 1)
    assert len(result) == 1
    assert result[0]["rank"] == 1
    assert result[0]["id"] == "p1"


def test_perturb_top_k_is_pure():
    original = [dict(p) for p in PASSAGES]
    perturb_top_k(PASSAGES, 1)
    assert PASSAGES == original


def test_inject_distractor_passages_adds_marked_passages():
    pool = [{"id": "d1", "text": "unrelated", "rank": 1, "score": 0.1}]
    result = inject_distractor_passages(PASSAGES, pool, n=1, seed=0)
    assert len(result) == len(PASSAGES) + 1
    distractors = [p for p in result if p.get("is_distractor")]
    assert len(distractors) == 1


def test_inject_distractor_passages_deterministic_given_seed():
    pool = [{"id": f"d{i}", "text": f"distractor {i}", "rank": 1, "score": 0.1} for i in range(5)]
    a = inject_distractor_passages(PASSAGES, pool, n=2, seed=7)
    b = inject_distractor_passages(PASSAGES, pool, n=2, seed=7)
    assert [p["id"] for p in a] == [p["id"] for p in b]


def test_remove_bridge_evidence_removes_matching_title():
    result = remove_bridge_evidence(PASSAGES, ["Einstein"])
    assert len(result) == 1
    assert result[0]["title"] == "Nobel"


def test_remove_bridge_evidence_requires_nonempty_titles():
    with pytest.raises(ValueError):
        remove_bridge_evidence(PASSAGES, [])


def test_inject_entity_alias_noise_changes_text_at_full_rate():
    result = inject_entity_alias_noise(PASSAGES, seed=1, noise_rate=1.0)
    assert any(r["text"] != p["text"] for r, p in zip(result, PASSAGES))


def test_inject_entity_alias_noise_zero_rate_is_noop():
    result = inject_entity_alias_noise(PASSAGES, seed=1, noise_rate=0.0)
    assert [r["text"] for r in result] == [p["text"] for p in PASSAGES]


def test_corrupt_numeric_and_date_mentions_changes_values_at_full_rate():
    result = corrupt_numeric_and_date_mentions(PASSAGES, seed=2, corruption_rate=1.0)
    assert any(r["text"] != p["text"] for r, p in zip(result, PASSAGES))


def test_corrupt_numeric_and_date_mentions_zero_rate_is_noop():
    result = corrupt_numeric_and_date_mentions(PASSAGES, seed=2, corruption_rate=0.0)
    assert [r["text"] for r in result] == [p["text"] for p in PASSAGES]
