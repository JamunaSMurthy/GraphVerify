"""Tests for graphverify/claim_decomposer.py."""
from __future__ import annotations

from graphverify.claim_decomposer import ClaimDecomposer


def test_decompose_uses_llm_claims(fake_llm):
    decomposer = ClaimDecomposer(fake_llm)
    claims = decomposer.decompose("Einstein won the Nobel Prize. He was born in Ulm.")
    assert len(claims) == 2
    assert "Einstein won the Nobel Prize." in claims


def test_decompose_empty_answer_returns_empty_list(fake_llm):
    decomposer = ClaimDecomposer(fake_llm)
    assert decomposer.decompose("") == []
    assert decomposer.decompose("   ") == []


def test_decompose_falls_back_to_sentence_split_on_llm_failure():
    class BrokenLLM:
        def chat_json(self, messages):
            return None  # simulates an unparseable LLM response

    decomposer = ClaimDecomposer(BrokenLLM())
    claims = decomposer.decompose("Einstein won the Nobel Prize. He was born in Ulm.")
    assert len(claims) == 2
