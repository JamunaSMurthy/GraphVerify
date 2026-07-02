"""Tests for experiments/_methods.py."""
from __future__ import annotations

import pytest

from baselines import BASELINE_REGISTRY
from experiments._methods import ALL_METHOD_NAMES, GRAPHVERIFY_METHOD_NAMES, build_method
from graphverify.config import GraphVerifyConfig
from graphverify.verifier import GraphVerify, HybridGraphVerify
from tests.fakes import FakeLLMClient


def test_all_method_names_includes_graphverify_and_every_baseline():
    assert set(GRAPHVERIFY_METHOD_NAMES) == {"graphverify_score", "graphverify_hybrid"}
    assert set(ALL_METHOD_NAMES) == set(GRAPHVERIFY_METHOD_NAMES) | set(BASELINE_REGISTRY)


def test_build_method_graphverify_score_uses_score_only_verdict_mode():
    method = build_method("graphverify_score", FakeLLMClient())
    assert isinstance(method._verifier, GraphVerify)
    assert not isinstance(method._verifier, HybridGraphVerify)
    assert method._verifier.cfg.verdict_mode == "score_only"


def test_build_method_graphverify_hybrid_uses_hybrid_verdict_mode():
    method = build_method("graphverify_hybrid", FakeLLMClient())
    assert isinstance(method._verifier, HybridGraphVerify)
    assert method._verifier.cfg.verdict_mode == "hybrid_llm"


def test_build_method_preserves_other_cfg_fields():
    cfg = GraphVerifyConfig(support_threshold=0.77, embed_model="custom-model")
    method = build_method("graphverify_score", FakeLLMClient(), cfg)
    assert method._verifier.cfg.support_threshold == 0.77
    assert method._verifier.cfg.embed_model == "custom-model"


@pytest.mark.parametrize("name", sorted(BASELINE_REGISTRY))
def test_build_method_constructs_every_baseline(name):
    method = build_method(name, FakeLLMClient())
    assert method is not None


def test_build_method_unknown_name_raises():
    with pytest.raises(ValueError):
        build_method("not_a_real_method", FakeLLMClient())


def test_uniform_verify_interface_across_all_methods(sample_query, sample_passages):
    claims = ["Albert Einstein was born in Ulm."]
    for name in ALL_METHOD_NAMES:
        method = build_method(name, FakeLLMClient())
        records = method.verify(sample_query, sample_passages, claims)
        assert isinstance(records, list)
        assert len(records) == 1
        assert "verdict" in records[0]
