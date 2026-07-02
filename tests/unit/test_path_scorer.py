"""Tests for graphverify/path_scorer.py."""
from __future__ import annotations

from graphverify.path_scorer import PathScorer


def _edge(src_label, relation, dst_label, confidence=0.9, rank=1):
    return {
        "src_label": src_label, "dst_label": dst_label, "relation": relation,
        "provenance": {"confidence": confidence, "retriever_rank": rank},
    }


def test_score_path_empty_returns_zero():
    scorer = PathScorer()
    scored = scorer.score_path([], "A", "rel", "B")
    assert scored.score == 0.0
    assert scored.head_match == "none"


def test_score_path_exact_match_scores_high():
    scorer = PathScorer()
    edges = [_edge("Einstein", "birthPlace", "Ulm")]
    scored = scorer.score_path(edges, "Einstein", "birthPlace", "Ulm")
    assert scored.head_match == "exact"
    assert scored.tail_match == "exact"
    assert scored.head_score == 1.0
    assert scored.tail_score == 1.0
    assert scored.score > 0.8


def test_score_path_weights_sum_contributions():
    scorer = PathScorer(lambda_head=0.3, lambda_rel=0.25, lambda_tail=0.3, lambda_prov=0.15)
    edges = [_edge("Einstein", "birthPlace", "Ulm", confidence=1.0, rank=1)]
    scored = scorer.score_path(edges, "Einstein", "birthPlace", "Ulm")
    expected = 0.3 * scored.head_score + 0.25 * scored.rel_score + 0.3 * scored.tail_score + 0.15 * scored.prov_score
    assert abs(scored.score - expected) < 1e-9


def test_exact_only_mode_rejects_partial_entity_match():
    scorer = PathScorer(match_mode="exact_only")
    edges = [_edge("Albert Einstein", "birthPlace", "Ulm")]
    scored = scorer.score_path(edges, "Einstein", "birthPlace", "Ulm")
    # "Einstein" != "Albert Einstein" exactly, and exact_only skips the alias tier
    assert scored.head_match == "none"
    assert scored.head_score == 0.0


def test_provenance_confidence_decays_with_rank():
    scorer = PathScorer()
    high_rank_edges = [_edge("A", "r", "B", confidence=0.9, rank=1)]
    low_rank_edges = [_edge("A", "r", "B", confidence=0.9, rank=10)]
    high = scorer.score_path(high_rank_edges, "A", "r", "B")
    low = scorer.score_path(low_rank_edges, "A", "r", "B")
    assert high.prov_score > low.prov_score
