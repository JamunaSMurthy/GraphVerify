"""
Tests for eval/ablation.py: every variant references a real GraphVerifyConfig
field (the fix for the previously-silent no-op bug), and run_variant raises
loudly on any variant that doesn't.
"""
from __future__ import annotations

from dataclasses import fields

import pytest

import graphverify.verifier as verifier_module
from eval.ablation import ABLATION_VARIANTS, AblationVariant, run_variant
from graphverify.config import GraphVerifyConfig
from tests.fakes import FakeLLMClient

_VALID_FIELDS = {f.name for f in fields(GraphVerifyConfig)}


def test_every_shipped_variant_uses_real_config_fields():
    for variant in ABLATION_VARIANTS:
        unknown = set(variant.kwargs) - _VALID_FIELDS
        assert not unknown, f"{variant.name} references unknown fields {unknown}"


def test_run_variant_raises_on_unknown_field():
    bad_variant = AblationVariant("bogus", {"this_field_does_not_exist": True})
    with pytest.raises(ValueError):
        run_variant(bad_variant, [], {}, GraphVerifyConfig())


def test_run_variant_applies_config_override(monkeypatch):
    monkeypatch.setattr(verifier_module, "LLMClient", lambda cfg: FakeLLMClient())
    samples = [{
        "id": "s1", "query": "q", "generated": "Einstein was born in Ulm.",
        "passages": [{"id": "p1", "text": "Einstein was born in Ulm.", "rank": 1, "score": 0.9}],
        "gold_verdict": "Supported", "gold_path": "",
    }]
    variant = AblationVariant("w/o contradiction detection", {"contradict_threshold": 1.01})
    metrics = run_variant(variant, samples, {}, GraphVerifyConfig())
    assert "claim_acc" in metrics


def test_run_variant_prefers_generated_over_gold_answer(monkeypatch):
    """
    Verifying the gold answer instead of the generated one would trivially
    bias toward "Supported" -- this asserts the fixed ordering (generated
    preferred, gold only as fallback).
    """
    seen_answers = []
    real_verify = verifier_module.GraphVerify.verify

    def spy_verify(self, query, passages, answer, **kwargs):
        seen_answers.append(answer)
        return real_verify(self, query, passages, answer, **kwargs)

    monkeypatch.setattr(verifier_module, "LLMClient", lambda cfg: FakeLLMClient())
    monkeypatch.setattr(verifier_module.GraphVerify, "verify", spy_verify)

    samples = [{
        "id": "s1", "query": "q", "answer": "GOLD ANSWER", "generated": "GENERATED ANSWER",
        "passages": [{"id": "p1", "text": "text", "rank": 1, "score": 0.9}],
        "gold_verdict": "Supported", "gold_path": "",
    }]
    run_variant(AblationVariant("Full GraphVerify", {}), samples, {}, GraphVerifyConfig())
    assert seen_answers == ["GENERATED ANSWER"]
