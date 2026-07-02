"""Tests for graphverify/relation_normalizer.py."""
from __future__ import annotations

from graphverify.relation_normalizer import RelationNormalizer


def test_exact_canonical_match():
    rn = RelationNormalizer()
    canon, score = rn.normalize("birthPlace")
    assert canon == "birthPlace"
    assert score == 1.0


def test_alias_match():
    rn = RelationNormalizer()
    canon, score = rn.normalize("born in")
    assert canon == "birthPlace"
    assert score == 1.0


def test_alias_match_case_insensitive():
    rn = RelationNormalizer()
    canon, _ = rn.normalize("BORN IN")
    assert canon == "birthPlace"


def test_unmatched_relation_returns_surface_unchanged():
    rn = RelationNormalizer()
    canon, score = rn.normalize("xyzzy nonrelation")
    assert canon == "xyzzy nonrelation"
    assert score == 0.0


def test_empty_surface_returns_unchanged():
    rn = RelationNormalizer()
    assert rn.normalize("") == ("", 0.0)


def test_disabled_normalizer_is_identity():
    rn = RelationNormalizer(disabled=True)
    canon, score = rn.normalize("born in")
    assert canon == "born in"  # NOT canonicalized to "birthPlace"
    assert score == 1.0
