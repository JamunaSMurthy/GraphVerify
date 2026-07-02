"""Tests for graphverify/verdict_assigner.py."""
from __future__ import annotations

from graphverify.path_scorer import ScoredPath
from graphverify.verdict_assigner import (
    VERDICT_CONTRADICTORY,
    VERDICT_SUPPORTED,
    VERDICT_UNSUPPORTED,
    VerdictAssigner,
    record_to_dict,
)


def _path(score):
    return ScoredPath(path_edges=[{"src_label": "A", "relation": "r", "dst_label": "B"}],
                       score=score, head_score=1.0, rel_score=1.0, tail_score=1.0, prov_score=1.0,
                       head_match="exact", tail_match="exact")


def test_unlinked_triple_is_unsupported():
    assigner = VerdictAssigner()
    rec = assigner.assign("claim", None, "rel", None, [], [], triple_linked=False)
    assert rec.verdict == VERDICT_UNSUPPORTED
    assert rec.best_path is None


def test_support_path_above_threshold_is_supported():
    assigner = VerdictAssigner(support_threshold=0.6, contradict_threshold=0.55)
    rec = assigner.assign("claim", "A", "rel", "B", [_path(0.8)], [], triple_linked=True)
    assert rec.verdict == VERDICT_SUPPORTED
    assert rec.best_path is not None
    assert rec.path_type == "support"


def test_conflict_path_above_threshold_is_contradictory():
    assigner = VerdictAssigner(support_threshold=0.6, contradict_threshold=0.55)
    rec = assigner.assign("claim", "A", "rel", "B", [], [_path(0.7)], triple_linked=True)
    assert rec.verdict == VERDICT_CONTRADICTORY
    assert rec.path_type == "conflict"


def test_contradiction_wins_when_both_paths_clear_threshold():
    """Contradiction must be checked before support -- see module docstring."""
    assigner = VerdictAssigner(support_threshold=0.6, contradict_threshold=0.55)
    rec = assigner.assign("claim", "A", "rel", "B", [_path(0.9)], [_path(0.9)], triple_linked=True)
    assert rec.verdict == VERDICT_CONTRADICTORY


def test_below_both_thresholds_is_unsupported():
    assigner = VerdictAssigner(support_threshold=0.6, contradict_threshold=0.55)
    rec = assigner.assign("claim", "A", "rel", "B", [_path(0.3)], [_path(0.2)], triple_linked=True)
    assert rec.verdict == VERDICT_UNSUPPORTED


def test_verdict_from_scores_matches_assign():
    assigner = VerdictAssigner(support_threshold=0.6, contradict_threshold=0.55)
    assert assigner.verdict_from_scores(0.8, 0.0) == VERDICT_SUPPORTED
    assert assigner.verdict_from_scores(0.0, 0.7) == VERDICT_CONTRADICTORY
    assert assigner.verdict_from_scores(0.9, 0.9) == VERDICT_CONTRADICTORY
    assert assigner.verdict_from_scores(0.1, 0.1) == VERDICT_UNSUPPORTED


def test_record_to_dict_strips_provenance_from_edge_dicts():
    assigner = VerdictAssigner(support_threshold=0.6, contradict_threshold=0.55)
    path = ScoredPath(
        path_edges=[{"src_label": "A", "relation": "r", "dst_label": "B", "provenance": {"secret": "x"}}],
        score=0.8, head_score=1, rel_score=1, tail_score=1, prov_score=1, head_match="exact", tail_match="exact",
    )
    rec = assigner.assign("claim", "A", "r", "B", [path], [], triple_linked=True)
    d = record_to_dict(rec)
    assert "provenance" not in d["best_path"][0]
    assert d["best_path"][0]["src_label"] == "A"


def test_record_to_dict_passes_through_string_paths_unchanged():
    """FIRE/CiteFix-style records store best_path as a list of passage id strings, not edge dicts."""
    assigner = VerdictAssigner()
    rec = assigner.assign("claim", "A", "r", "B", [_path(0.9)], [], triple_linked=True)
    rec.best_path = ["p1", "p2"]
    d = record_to_dict(rec)
    assert d["best_path"] == ["p1", "p2"]
