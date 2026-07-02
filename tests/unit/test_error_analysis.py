"""Tests for eval/error_analysis.py."""
from __future__ import annotations

from eval.error_analysis import (
    ErrorSample,
    load_error_samples_csv,
    sample_errors,
    save_error_samples_csv,
    summarize_error_categories,
)


def _record(id_, verdict, gold, triple_linked=True, path_type="support", claim="claim text"):
    return {"id": id_, "claim": claim, "verdict": verdict, "gold_verdict": gold,
            "triple_linked": triple_linked, "path_type": path_type, "reliability": 0.5, "verdict_mode": "score_only"}


def test_sample_errors_only_includes_mispredictions():
    records = [
        _record("1", "Supported", "Supported"),   # correct, excluded
        _record("2", "Unsupported", "Supported"),  # error
        _record("3", "Contradictory", "Supported"),  # error
    ]
    samples = sample_errors(records, "demo", n=10, seed=0)
    assert len(samples) == 2
    assert all(s.predicted_verdict != s.gold_verdict for s in samples)


def test_sample_errors_respects_n_cap():
    records = [_record(str(i), "Unsupported", "Supported") for i in range(20)]
    samples = sample_errors(records, "demo", n=5, seed=0)
    assert len(samples) == 5


def test_sample_errors_is_deterministic_given_seed():
    records = [_record(str(i), "Unsupported", "Supported") for i in range(20)]
    a = sample_errors(records, "demo", n=5, seed=42)
    b = sample_errors(records, "demo", n=5, seed=42)
    assert [s.item_id for s in a] == [s.item_id for s in b]


def test_suggest_category_entity_linking_error():
    records = [_record("1", "Unsupported", "Supported", triple_linked=False)]
    samples = sample_errors(records, "demo", n=10, seed=0)
    assert samples[0].suggested_category == "entity_linking_error"


def test_suggest_category_missing_retrieval_evidence():
    records = [_record("1", "Unsupported", "Supported", triple_linked=True, path_type="none")]
    samples = sample_errors(records, "demo", n=10, seed=0)
    assert samples[0].suggested_category == "missing_retrieval_evidence"


def test_suggest_category_false_conflict():
    records = [_record("1", "Contradictory", "Supported")]
    samples = sample_errors(records, "demo", n=10, seed=0)
    assert samples[0].suggested_category == "false_conflict"


def test_csv_roundtrip(tmp_path):
    samples = [
        ErrorSample(item_id="1", dataset="demo", claim="c", predicted_verdict="Unsupported",
                    gold_verdict="Supported", triple_linked=True, path_type="none",
                    reliability=0.1, verdict_mode="score_only", suggested_category="missing_retrieval_evidence"),
    ]
    path = str(tmp_path / "errors.csv")
    save_error_samples_csv(samples, path)
    loaded = load_error_samples_csv(path)
    assert len(loaded) == 1
    assert loaded[0].item_id == "1"
    assert loaded[0].triple_linked is True


def test_summarize_prefers_human_category_over_suggested():
    samples = [
        ErrorSample(item_id="1", dataset="d", claim="c", predicted_verdict="Unsupported", gold_verdict="Supported",
                    triple_linked=True, path_type="none", reliability=0.0, verdict_mode="score_only",
                    suggested_category="missing_retrieval_evidence", human_category="claim_decomposition_error"),
        ErrorSample(item_id="2", dataset="d", claim="c", predicted_verdict="Unsupported", gold_verdict="Supported",
                    triple_linked=True, path_type="none", reliability=0.0, verdict_mode="score_only",
                    suggested_category="missing_retrieval_evidence"),
    ]
    counts = summarize_error_categories(samples)
    assert counts == {"claim_decomposition_error": 1, "missing_retrieval_evidence": 1}
