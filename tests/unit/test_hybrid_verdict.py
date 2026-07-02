"""Tests for graphverify/hybrid_verdict.py."""
from __future__ import annotations

from graphverify.hybrid_verdict import HybridVerdictHead
from graphverify.path_scorer import ScoredPath
from graphverify.verdict_assigner import VerificationRecord
from tests.fakes import FakeLLMClient


def _rule_based(verdict="Supported", reliability=0.65, triple_linked=True):
    return VerificationRecord(
        claim="claim", head="A", relation="rel", tail="B",
        verdict=verdict, best_path=None, reliability=reliability,
        support_score=reliability, contradict_score=0.0,
        path_type="support", triple_linked=triple_linked,
    )


def _path(score=0.7):
    return ScoredPath(path_edges=[{"src_label": "A", "relation": "rel", "dst_label": "B"}],
                       score=score, head_score=1, rel_score=1, tail_score=1, prov_score=1,
                       head_match="exact", tail_match="exact")


def test_unlinked_triple_short_circuits_without_llm_call():
    class ExplodingLLM:
        def chat_json(self, messages):
            raise AssertionError("should not be called for an unlinked triple")

    head = HybridVerdictHead(ExplodingLLM())
    decision = head.decide("claim", None, "rel", None, _rule_based(triple_linked=False), [], [])
    assert decision.verdict == "Unsupported"
    assert decision.overrode_rule_based is False


def test_default_fake_llm_confirms_rule_based_prior():
    head = HybridVerdictHead(FakeLLMClient())
    rule_based = _rule_based(verdict="Supported")
    decision = head.decide("claim", "A", "rel", "B", rule_based, [_path(0.7)], [])
    assert decision.verdict == "Supported"
    assert decision.overrode_rule_based is False


def test_override_via_scripted_response():
    llm = FakeLLMClient(overrides=[
        (lambda system, user: "verdict head of GraphVerify-Hybrid" in system,
         {"verdict": "Contradictory", "confidence": 0.95, "rationale": "path text shows a direct conflict"}),
    ])
    head = HybridVerdictHead(llm)
    rule_based = _rule_based(verdict="Supported")
    decision = head.decide("claim", "A", "rel", "B", rule_based, [_path(0.55)], [_path(0.6)])
    assert decision.verdict == "Contradictory"
    assert decision.overrode_rule_based is True
    assert decision.confidence == 0.95


def test_unparseable_response_falls_back_to_rule_based_prior():
    class BrokenLLM:
        def chat_json(self, messages):
            return None

    head = HybridVerdictHead(BrokenLLM())
    rule_based = _rule_based(verdict="Supported", reliability=0.7)
    decision = head.decide("claim", "A", "rel", "B", rule_based, [_path(0.7)], [])
    assert decision.verdict == "Supported"
    assert decision.confidence == 0.7
    assert decision.overrode_rule_based is False


def test_invalid_verdict_string_falls_back_to_rule_based():
    llm = FakeLLMClient(overrides=[
        (lambda system, user: "verdict head of GraphVerify-Hybrid" in system,
         {"verdict": "MaybeSupported", "confidence": 0.5, "rationale": "..."}),
    ])
    head = HybridVerdictHead(llm)
    rule_based = _rule_based(verdict="Unsupported")
    decision = head.decide("claim", "A", "rel", "B", rule_based, [], [])
    assert decision.verdict == "Unsupported"
