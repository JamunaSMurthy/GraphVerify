"""Tests for eval/contradiction_taxonomy.py."""
from __future__ import annotations

from eval.contradiction_taxonomy import classify_contradiction, contradiction_taxonomy_breakdown


def _edge(src, relation, dst):
    return {"src_label": src, "relation": relation, "dst_label": dst}


def test_non_contradictory_record_returns_none():
    assert classify_contradiction({"verdict": "Supported"}) is None


def test_no_path_returns_none():
    assert classify_contradiction({"verdict": "Contradictory", "best_path": None}) is None
    assert classify_contradiction({"verdict": "Contradictory", "best_path": ["p1", "p2"]}) is None  # strings, not edge dicts


def test_functional_relation_conflict_classified_as_entity_functional():
    rec = {
        "verdict": "Contradictory", "relation": "birthPlace", "tail": "Paris",
        "best_path": [_edge("Einstein", "birthPlace", "London")],
    }
    assert classify_contradiction(rec) == "entity_functional"


def test_relation_mismatch_detected_before_value_conflict():
    rec = {
        "verdict": "Contradictory", "relation": "birthPlace", "tail": "Paris",
        "best_path": [_edge("Einstein", "deathPlace", "Paris")],
    }
    assert classify_contradiction(rec) == "relation_mismatch"


def test_multi_hop_when_no_specific_rule_fires_and_path_has_multiple_edges():
    rec = {
        "verdict": "Contradictory", "relation": "genre", "tail": "Comedy",
        "best_path": [_edge("Movie", "genre", "Intermediate"), _edge("Intermediate", "genre", "Drama")],
    }
    assert classify_contradiction(rec) == "multi_hop"


def test_unknown_when_single_edge_and_no_rule_fires():
    rec = {
        "verdict": "Contradictory", "relation": "genre", "tail": "Comedy",
        "best_path": [_edge("Movie", "genre", "Drama")],
    }
    assert classify_contradiction(rec) == "unknown"


def test_breakdown_counts_only_classifiable_contradictions():
    records = [
        {"verdict": "Supported"},
        {"verdict": "Contradictory", "best_path": None},  # unclassifiable
        {"verdict": "Contradictory", "relation": "birthPlace", "tail": "Paris",
         "best_path": [_edge("A", "birthPlace", "London")]},
    ]
    breakdown = contradiction_taxonomy_breakdown(records)
    assert breakdown == {"entity_functional": 1}
