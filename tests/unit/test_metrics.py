"""Tests for eval/metrics.py."""
from __future__ import annotations

import pytest

from eval.metrics import (
    claim_accuracy,
    compute_all_metrics,
    contradiction_f1,
    hallucination_auroc_auprc,
    macro_f1,
    path_correctness,
    per_class_ece,
    run_bootstrap,
    supported_f1,
    unsupported_f1,
)

PREDS = ["Supported", "Unsupported", "Contradictory", "Supported", "Contradictory"]
GOLDS = ["Supported", "Unsupported", "Supported", "Supported", "Contradictory"]


def test_claim_accuracy():
    assert claim_accuracy(["Supported"], ["Supported"]) == 100.0
    assert claim_accuracy(["Supported"], ["Unsupported"]) == 0.0
    assert claim_accuracy(PREDS, GOLDS) == pytest.approx(80.0)


def test_per_class_f1_perfect_and_zero():
    assert unsupported_f1(["Unsupported"], ["Unsupported"]) == 100.0
    assert contradiction_f1(["Supported"], ["Contradictory"]) == 0.0


def test_class_f1_with_no_positive_golds_is_zero():
    assert contradiction_f1(["Contradictory"], ["Supported"]) == 0.0


def test_macro_f1_averages_three_classes():
    m = macro_f1(PREDS, GOLDS)
    expected = (supported_f1(PREDS, GOLDS) + unsupported_f1(PREDS, GOLDS) + contradiction_f1(PREDS, GOLDS)) / 3
    assert m == pytest.approx(expected)


def test_path_correctness_exact_and_token_f1():
    assert path_correctness(["A -> B"], ["A -> B"], method="exact") == 100.0
    assert path_correctness(["A -> B -> C"], ["A -> B"], method="f1_token") > 0.0
    assert path_correctness([None], [""]) == 100.0  # both empty -> compatible
    assert path_correctness([None], ["A -> B"]) == 0.0


def test_compute_all_metrics_returns_expected_keys():
    result = compute_all_metrics(PREDS, GOLDS, rel_scores=[0.9, 0.8, 0.5, 0.9, 0.6])
    for key in ("claim_acc", "supp_f1", "unsupp_f1", "contr_f1", "macro_f1", "ece"):
        assert key in result


def test_per_class_ece_includes_overall_and_present_classes():
    result = per_class_ece(PREDS, GOLDS, [0.9, 0.8, 0.5, 0.9, 0.6])
    assert "overall" in result
    assert set(result) - {"overall"} <= {"Supported", "Unsupported", "Contradictory"}


def test_hallucination_auroc_auprc_perfect_separation():
    scores = [0.9, 0.9, 0.1, 0.1]  # high reliability -> not hallucinated
    is_halluc = [0, 0, 1, 1]
    result = hallucination_auroc_auprc(scores, is_halluc)
    assert result["auroc"] == pytest.approx(1.0)
    assert result["auprc"] == pytest.approx(1.0)


def test_hallucination_auroc_auprc_undefined_with_single_class():
    result = hallucination_auroc_auprc([0.9, 0.1], [0, 0])
    assert result["auroc"] != result["auroc"]  # NaN != NaN


def test_run_bootstrap_plain_matches_point_estimate():
    point, lo, hi = run_bootstrap(PREDS, GOLDS, metric="claim_acc", n_boot=200, seed=0)
    assert point == claim_accuracy(PREDS, GOLDS)
    assert lo <= point <= hi


def test_run_bootstrap_clustered_produces_valid_interval():
    cluster_ids = ["a", "a", "b", "b", "c"]
    point, lo, hi = run_bootstrap(PREDS, GOLDS, metric="claim_acc", cluster_ids=cluster_ids, n_boot=200, seed=0)
    assert lo <= point <= hi
    assert 0.0 <= lo and hi <= 100.0
