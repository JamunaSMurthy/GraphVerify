"""Tests for eval/coverage_reliability.py."""
from __future__ import annotations

import pytest

from eval.coverage_reliability import (
    accept_answer,
    coverage_reliability_report,
    exact_match_score,
    f1_score,
    hallucination_precision_recall,
    is_claim_acceptable,
    normalize_answer,
)


def test_is_claim_acceptable_contradictory_always_rejects():
    assert is_claim_acceptable("Contradictory", 0.99, 0.5) is False


def test_is_claim_acceptable_unsupported_always_accepts():
    assert is_claim_acceptable("Unsupported", 0.0, 0.9) is True


def test_is_claim_acceptable_supported_gated_by_threshold():
    assert is_claim_acceptable("Supported", 0.8, 0.7) is True
    assert is_claim_acceptable("Supported", 0.6, 0.7) is False


def test_accept_answer_all_claims_must_pass():
    good = [{"verdict": "Supported", "reliability": 0.9}, {"verdict": "Unsupported", "reliability": 0.0}]
    assert accept_answer(good, 0.8) is True

    bad = [{"verdict": "Supported", "reliability": 0.9}, {"verdict": "Contradictory", "reliability": 0.9}]
    assert accept_answer(bad, 0.8) is False


def test_accept_answer_empty_claims_is_vacuously_true():
    assert accept_answer([], 0.5) is True


def test_normalize_answer_strips_articles_punctuation_and_case():
    assert normalize_answer("The Paris City.") == normalize_answer("paris city")


def test_exact_match_score():
    assert exact_match_score("Paris", "the Paris") == 1.0
    assert exact_match_score("Paris", "London") == 0.0


def test_f1_score_partial_overlap():
    score = f1_score("big red dog", "big blue dog")
    assert 0.0 < score < 1.0


def test_f1_score_empty_vs_empty_is_one():
    assert f1_score("", "") == 1.0


def test_coverage_reliability_report_monotonic_acceptance():
    answers = [
        {"generated": "Paris", "gold_answer": "Paris",
         "claim_records": [{"verdict": "Supported", "reliability": 0.95}]},
        {"generated": "London", "gold_answer": "Paris",
         "claim_records": [{"verdict": "Supported", "reliability": 0.55}]},
    ]
    results = coverage_reliability_report(answers, thresholds=[0.5, 0.9])
    low_thresh, high_thresh = results
    # raising the threshold should never increase the accepted percentage
    assert high_thresh.accepted_pct <= low_thresh.accepted_pct


def test_hallucination_precision_recall_perfect_case():
    scores = [0.9, 0.9, 0.1, 0.1]
    is_halluc = [0, 0, 1, 1]
    result = hallucination_precision_recall(scores, is_halluc, threshold=0.5)
    assert result["precision"] == pytest.approx(100.0)
    assert result["recall"] == pytest.approx(100.0)
