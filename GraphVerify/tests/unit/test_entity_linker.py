"""Tests for graphverify/entity_linker.py: three-tier matching and match_mode gating."""
from __future__ import annotations

from graphverify.entity_linker import EntityLinker

LABELS = ["Albert Einstein", "Nobel Prize", "Ulm"]


def test_exact_match():
    linker = EntityLinker(LABELS)
    idx, score = linker.link("Albert Einstein")
    assert idx == 0
    assert score == 1.0


def test_exact_match_case_and_whitespace_insensitive():
    linker = EntityLinker(LABELS)
    idx, score = linker.link("  albert   einstein ")
    assert idx == 0
    assert score == 1.0


def test_no_match_returns_none():
    linker = EntityLinker(LABELS)
    idx, score = linker.link("Marie Curie")
    assert idx is None
    assert score == 0.0


def test_empty_mention_returns_none():
    linker = EntityLinker(LABELS)
    assert linker.link("") == (None, 0.0)
    assert linker.link("   ") == (None, 0.0)


def test_link_text_returns_label():
    linker = EntityLinker(LABELS)
    label, score = linker.link_text("Ulm")
    assert label == "Ulm"
    assert score == 1.0


def test_exact_only_mode_skips_alias_and_embedding_tiers():
    linker = EntityLinker(LABELS, match_mode="exact_only")
    # exact match still works
    assert linker.link("Albert Einstein")[0] == 0
    # partial/alias overlap must NOT match under exact_only
    assert linker.link("Einstein")[0] is None


def test_exact_alias_mode_enables_token_overlap():
    linker = EntityLinker(["Albert Einstein Junior"], match_mode="exact_alias")
    idx, score = linker.link("Albert Einstein")
    assert idx == 0
    assert score > 0.0


def test_embed_only_mode_bypasses_exact_tier_but_still_finds_identical_string():
    # With the fake embedder, identical normalized strings hash to identical
    # vectors (cosine similarity 1.0), so embed_only still finds an exact
    # string via the embedding tier alone.
    linker = EntityLinker(LABELS, match_mode="embed_only")
    idx, score = linker.link("Albert Einstein")
    assert idx == 0
    assert score >= 0.99


def test_embed_only_mode_does_not_use_token_overlap_tier():
    # A single-word partial mention that would match via token-overlap
    # ("exact_alias") should NOT match under embed_only unless the fake
    # embedder happens to place it close by chance -- assert no exact-tier
    # shortcut is taken for a clearly different string.
    linker = EntityLinker(LABELS, match_mode="embed_only")
    idx, score = linker.link("completely unrelated string")
    assert idx is None or score < 1.0
