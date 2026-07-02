"""Tests for graphverify/text_evidence.py."""
from __future__ import annotations

from graphverify.text_evidence import text_entailment_verdict


def test_no_passages_returns_unsupported(fake_llm):
    result = text_entailment_verdict(fake_llm, "Some claim.", [])
    assert result.verdict == "Unsupported"
    assert result.confidence == 0.0


def test_high_overlap_evidence_supports_claim(fake_llm):
    passages = [{"id": "p1", "text": "Einstein won the Nobel Prize in 1921."}]
    result = text_entailment_verdict(fake_llm, "Einstein won the Nobel Prize in 1921.", passages)
    assert result.verdict == "Supported"


def test_unrelated_evidence_is_unsupported(fake_llm):
    passages = [{"id": "p1", "text": "Bananas are a good source of potassium."}]
    result = text_entailment_verdict(fake_llm, "Einstein won the Nobel Prize in 1921.", passages)
    assert result.verdict == "Unsupported"


def test_unparseable_llm_response_is_unsupported_not_contradictory():
    class BrokenLLM:
        def chat_json(self, messages):
            return None

    passages = [{"id": "p1", "text": "anything"}]
    result = text_entailment_verdict(BrokenLLM(), "claim", passages)
    assert result.verdict == "Unsupported"
    assert result.confidence == 0.0
