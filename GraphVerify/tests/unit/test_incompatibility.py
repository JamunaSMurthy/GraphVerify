"""Tests for graphverify/incompatibility.py: all four conflict rules and the classifier."""
from __future__ import annotations

from graphverify.incompatibility import classify_incompatibility, is_incompatible


def test_equal_values_are_compatible():
    assert classify_incompatibility("Paris", "Paris", "birthPlace") is None
    assert is_incompatible("Paris", "Paris", "birthPlace") is False


def test_functional_relation_conflict():
    assert classify_incompatibility("Paris", "London", "birthPlace") == "entity_functional"
    assert is_incompatible("Paris", "London", "birthPlace") is True


def test_temporal_relation_conflict():
    # "year" is in TEMPORAL_RELATIONS but not FUNCTIONAL_RELATIONS, so this
    # exercises the temporal rule specifically (some relations, like
    # "releaseDate", are in both lists and would hit the functional rule
    # first per the priority order documented in incompatibility.py).
    assert classify_incompatibility("1990", "1995", "year") == "temporal"


def test_temporal_relation_same_year_is_compatible():
    assert classify_incompatibility("1990", "1990", "year") is None


def test_numeric_conflict_beyond_tolerance():
    # 100 vs 200 is a 100% relative difference, well beyond the 5% default tolerance,
    # and "population" is not itself a functional/temporal relation trigger... note
    # population IS in FUNCTIONAL_RELATIONS in this codebase's config, so use a
    # relation outside both FUNCTIONAL_RELATIONS and TEMPORAL_RELATIONS.
    assert classify_incompatibility("100", "200", "cost") == "numeric"


def test_numeric_within_tolerance_is_compatible():
    assert classify_incompatibility("100", "102", "cost") is None


def test_mutually_exclusive_labels():
    assert classify_incompatibility("supports", "refutes", "verdict") == "mutually_exclusive"
    assert classify_incompatibility("true", "false", "verdict") == "mutually_exclusive"


def test_no_rule_fires_returns_none():
    assert classify_incompatibility("blue", "green", "hasProperty") is None


def test_empty_values_are_compatible():
    assert classify_incompatibility("", "London", "birthPlace") is None
    assert classify_incompatibility("Paris", "", "birthPlace") is None


def test_is_incompatible_matches_classify_incompatibility():
    cases = [
        ("Paris", "London", "birthPlace"),
        ("100", "200", "cost"),
        ("blue", "green", "hasProperty"),
    ]
    for claim_tail, path_tail, relation in cases:
        assert is_incompatible(claim_tail, path_tail, relation) == (
            classify_incompatibility(claim_tail, path_tail, relation) is not None
        )
