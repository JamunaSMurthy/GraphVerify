"""
End-to-end smoke test: runs GraphVerify.verify(), HybridGraphVerify.verify(),
and every baseline in the registry against one fixture query/passage/answer
using the fake LLM/embedder doubles, asserting each returns well-formed
claim-level verdict records. This is the single test that proves the full
11-method registry is wired together correctly, end to end, offline.
"""
from __future__ import annotations

import pytest

from baselines import BASELINE_REGISTRY
from experiments._methods import ALL_METHOD_NAMES, build_method
from graphverify.config import GraphVerifyConfig
from graphverify.verifier import build_graphverify
from tests.fakes import FakeLLMClient

QUERY = "Who designed the Eiffel Tower and where is it located?"
PASSAGES = [
    {"id": "p1", "rank": 1, "score": 0.95, "text": "The Eiffel Tower was designed by Gustave Eiffel and completed in 1889."},
    {"id": "p2", "rank": 2, "score": 0.80, "text": "The Eiffel Tower is located in Paris, France."},
]
ANSWER = "The Eiffel Tower was designed by Gustave Eiffel. The Eiffel Tower is located in Paris."


def _assert_well_formed(records):
    assert isinstance(records, list)
    assert len(records) >= 1
    for rec in records:
        assert rec["verdict"] in ("Supported", "Unsupported", "Contradictory")
        assert 0.0 <= rec["reliability"] <= 1.0
        assert isinstance(rec["claim"], str) and rec["claim"]


def test_graphverify_score_end_to_end():
    gv = build_graphverify(GraphVerifyConfig(verdict_mode="score_only"), llm_client=FakeLLMClient())
    out = gv.verify(query=QUERY, passages=PASSAGES, answer=ANSWER)
    _assert_well_formed(out.records)
    assert all(r["verdict_mode"] == "score_only" for r in out.records)


def test_graphverify_hybrid_end_to_end():
    gv = build_graphverify(GraphVerifyConfig(verdict_mode="hybrid_llm"), llm_client=FakeLLMClient())
    out = gv.verify(query=QUERY, passages=PASSAGES, answer=ANSWER)
    _assert_well_formed(out.records)
    assert all(r["verdict_mode"] == "hybrid_llm" for r in out.records)


@pytest.mark.parametrize("name", sorted(BASELINE_REGISTRY))
def test_every_baseline_end_to_end(name):
    method = build_method(name, FakeLLMClient())
    claims = ["The Eiffel Tower was designed by Gustave Eiffel.", "The Eiffel Tower is located in Paris."]
    records = method.verify(QUERY, PASSAGES, claims)
    _assert_well_formed(records)


def test_full_registry_covers_eleven_methods():
    assert len(ALL_METHOD_NAMES) == 11
