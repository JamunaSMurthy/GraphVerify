"""Tests for baselines/: registry conformance and per-baseline sanity checks."""
from __future__ import annotations

import pytest

from baselines import BASELINE_REGISTRY
from graphverify.verdict_assigner import VerificationRecord
from tests.fakes import FakeLLMClient

VALID_CATEGORIES = {"native_posthoc", "adapted_graph_retrieval", "kg_fact_checking", "ablation_control"}


@pytest.mark.parametrize("name,cls", sorted(BASELINE_REGISTRY.items()))
def test_every_baseline_declares_required_metadata(name, cls):
    assert cls.name == name
    assert cls.citation, f"{name} must cite the method it approximates"
    assert cls.category in VALID_CATEGORIES
    assert isinstance(cls.uses_graph, bool)


@pytest.mark.parametrize("name,cls", sorted(BASELINE_REGISTRY.items()))
def test_every_baseline_verifies_claims_without_error(name, cls, sample_query, sample_passages):
    verifier = cls(FakeLLMClient())
    claims = ["Albert Einstein was born in Ulm, Germany."]
    records = verifier.verify_claims(sample_query, sample_passages, claims)
    assert len(records) == 1
    assert isinstance(records[0], VerificationRecord)
    assert records[0].verdict in ("Supported", "Unsupported", "Contradictory")
    assert records[0].verdict_mode == name


@pytest.mark.parametrize("name,cls", sorted(BASELINE_REGISTRY.items()))
def test_every_baseline_handles_empty_claims_list(name, cls, sample_query, sample_passages):
    verifier = cls(FakeLLMClient())
    assert verifier.verify_claims(sample_query, sample_passages, []) == []


@pytest.mark.parametrize("name,cls", sorted(BASELINE_REGISTRY.items()))
def test_every_baseline_handles_empty_passages(name, cls, sample_query):
    verifier = cls(FakeLLMClient())
    records = verifier.verify_claims(sample_query, [], ["Some claim."])
    assert len(records) == 1
    assert records[0].verdict == "Unsupported"


def test_base_verify_decomposes_when_claims_not_supplied(sample_query, sample_passages):
    from baselines.llm_text_verifier import LLMTextVerifier
    verifier = LLMTextVerifier(FakeLLMClient())
    records = verifier.verify(sample_query, sample_passages, "Einstein was born in Ulm. He won a prize.")
    assert len(records) == 2


def test_graphrag_adapted_and_hipporag_adapted_use_shared_verdict_head_types():
    """
    The adapted graph-retrieval baselines must reuse graphverify's own
    PathSearcher/VerdictAssigner rather than a separately implemented rule
    -- this is the fairness protocol's "shared verdict head" requirement.
    """
    from baselines.graphrag_adapted import GraphRAGAdaptedVerifier
    from baselines.hipporag_adapted import HippoRAGAdaptedVerifier
    from graphverify.verdict_assigner import VerdictAssigner

    for cls in (GraphRAGAdaptedVerifier, HippoRAGAdaptedVerifier):
        verifier = cls(FakeLLMClient())
        assert isinstance(verifier._verdict, VerdictAssigner)


def test_llm_text_verifier_uses_no_graph():
    from baselines.llm_text_verifier import LLMTextVerifier
    assert LLMTextVerifier.uses_graph is False
    assert LLMTextVerifier.category == "ablation_control"


def test_adapted_baselines_are_marked_adapted_not_native():
    from baselines.graphrag_adapted import GraphRAGAdaptedVerifier
    from baselines.hipporag_adapted import HippoRAGAdaptedVerifier
    assert GraphRAGAdaptedVerifier.category == "adapted_graph_retrieval"
    assert HippoRAGAdaptedVerifier.category == "adapted_graph_retrieval"
