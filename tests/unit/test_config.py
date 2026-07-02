"""Tests for graphverify/config.py: invariants and default values."""
from __future__ import annotations

from graphverify.config import (
    CONTRADICT_THRESHOLD,
    LAMBDA_HEAD,
    LAMBDA_PROV,
    LAMBDA_REL,
    LAMBDA_TAIL,
    SUPPORT_THRESHOLD,
    GraphVerifyConfig,
)


def test_lambda_weights_sum_to_one():
    assert abs((LAMBDA_HEAD + LAMBDA_REL + LAMBDA_TAIL + LAMBDA_PROV) - 1.0) < 1e-9


def test_default_config_matches_module_constants():
    cfg = GraphVerifyConfig()
    assert cfg.support_threshold == SUPPORT_THRESHOLD
    assert cfg.contradict_threshold == CONTRADICT_THRESHOLD
    assert cfg.verdict_mode == "score_only"
    assert cfg.evidence_mode == "hybrid"
    assert cfg.evidence_source == "retrieved_only"
    assert cfg.entity_match_mode == "exact_alias_embed"
    assert cfg.disable_claim_decomposition is False
    assert cfg.disable_relation_normalization is False


def test_config_is_overridable():
    cfg = GraphVerifyConfig(support_threshold=0.9, verdict_mode="hybrid_llm")
    assert cfg.support_threshold == 0.9
    assert cfg.verdict_mode == "hybrid_llm"


def test_seeds_default_factory_is_independent_per_instance():
    a = GraphVerifyConfig()
    b = GraphVerifyConfig()
    a.seeds.append(99)
    assert 99 not in b.seeds
