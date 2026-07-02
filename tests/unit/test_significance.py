"""Tests for eval/significance.py."""
from __future__ import annotations

from eval.significance import paired_bootstrap_significance


def test_identical_systems_have_zero_effect_size_and_p_near_one():
    preds = ["Supported", "Unsupported", "Contradictory", "Supported"]
    golds = ["Supported", "Unsupported", "Supported", "Supported"]
    result = paired_bootstrap_significance(preds, preds, golds, metric="claim_acc", n_boot=200, seed=0)
    assert result.effect_size == 0.0
    assert result.p_value == 1.0


def test_clearly_better_system_has_positive_effect_size():
    golds = ["Supported"] * 10
    preds_a = ["Supported"] * 10           # always correct
    preds_b = ["Unsupported"] * 10          # always wrong
    result = paired_bootstrap_significance(preds_a, preds_b, golds, metric="claim_acc", n_boot=200, seed=0)
    assert result.effect_size > 0
    assert result.system_a_score == 100.0
    assert result.system_b_score == 0.0


def test_clustered_resampling_runs_without_error():
    golds = ["Supported", "Supported", "Unsupported", "Unsupported"]
    preds_a = ["Supported", "Unsupported", "Unsupported", "Supported"]
    preds_b = ["Unsupported", "Unsupported", "Unsupported", "Unsupported"]
    cluster_ids = ["item1", "item1", "item2", "item2"]
    result = paired_bootstrap_significance(preds_a, preds_b, golds, metric="claim_acc", cluster_ids=cluster_ids, n_boot=100, seed=0)
    assert result.ci_low <= result.effect_size <= result.ci_high
