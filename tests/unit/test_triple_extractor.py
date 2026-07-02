"""Tests for graphverify/triple_extractor.py."""
from __future__ import annotations

from graphverify.entity_linker import EntityLinker
from graphverify.relation_normalizer import RelationNormalizer
from graphverify.triple_extractor import TripleExtractor


def test_extract_links_entities_present_in_graph(fake_llm):
    linker = EntityLinker(["Einstein", "1921"])
    rel_norm = RelationNormalizer()
    extractor = TripleExtractor(fake_llm, linker, rel_norm)

    triple = extractor.extract("Einstein won the Nobel Prize in 1921.")
    assert triple.linked is True
    assert triple.head == "Einstein"
    assert triple.tail == "1921"


def test_extract_unlinked_when_entities_not_in_graph(fake_llm):
    linker = EntityLinker(["Marie Curie"])  # neither head nor tail will be found
    rel_norm = RelationNormalizer()
    extractor = TripleExtractor(fake_llm, linker, rel_norm)

    triple = extractor.extract("Einstein won the Nobel Prize in 1921.")
    assert triple.linked is False


def test_extract_returns_unlinked_triple_on_llm_failure():
    class BrokenLLM:
        def chat_json(self, messages):
            return {}  # missing head/relation/tail keys

    linker = EntityLinker(["Einstein"])
    extractor = TripleExtractor(BrokenLLM(), linker, RelationNormalizer())
    triple = extractor.extract("Einstein won the Nobel Prize.")
    assert triple.linked is False
    assert triple.head is None
