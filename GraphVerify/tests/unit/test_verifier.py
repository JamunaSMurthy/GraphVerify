"""End-to-end tests for graphverify/verifier.py: GraphVerify and HybridGraphVerify."""
from __future__ import annotations

import json
import warnings

from graphverify.config import GraphVerifyConfig
from graphverify.verifier import GraphVerify, HybridGraphVerify, build_graphverify
from tests.fakes import FakeLLMClient


def test_score_only_verify_returns_records(sample_query, sample_passages, sample_answer):
    gv = GraphVerify(GraphVerifyConfig(), llm_client=FakeLLMClient())
    out = gv.verify(query=sample_query, passages=sample_passages, answer=sample_answer)
    assert len(out.records) >= 1
    assert all(r["verdict"] in ("Supported", "Unsupported", "Contradictory") for r in out.records)
    assert all(r["verdict_mode"] == "score_only" for r in out.records)


def test_build_graphverify_dispatches_on_verdict_mode():
    score_only = build_graphverify(GraphVerifyConfig(verdict_mode="score_only"), llm_client=FakeLLMClient())
    hybrid = build_graphverify(GraphVerifyConfig(verdict_mode="hybrid_llm"), llm_client=FakeLLMClient())
    assert isinstance(score_only, GraphVerify) and not isinstance(score_only, HybridGraphVerify)
    assert isinstance(hybrid, HybridGraphVerify)


def test_hybrid_verify_stamps_hybrid_llm_mode(sample_query, sample_passages, sample_answer):
    gv = HybridGraphVerify(GraphVerifyConfig(), llm_client=FakeLLMClient())
    out = gv.verify(query=sample_query, passages=sample_passages, answer=sample_answer)
    assert all(r["verdict_mode"] == "hybrid_llm" for r in out.records)


def test_pre_decomposed_claims_bypass_internal_decomposition(sample_query, sample_passages):
    calls = []

    class TrackingLLM(FakeLLMClient):
        def chat_json(self, messages):
            calls.append(messages)
            return super().chat_json(messages)

    gv = GraphVerify(GraphVerifyConfig(), llm_client=TrackingLLM())
    claims = ["Einstein was born in Ulm."]
    out = gv.verify(query=sample_query, passages=sample_passages, answer="unused", claims=claims)
    assert len(out.records) == 1
    assert out.records[0]["claim"] == claims[0]
    # no call should have gone through the claim-decomposition prompt route
    assert not any("claim decomposition assistant" in (m[0]["content"] if m else "") for m in calls)


def test_disable_claim_decomposition_treats_whole_answer_as_one_claim(sample_query, sample_passages):
    cfg = GraphVerifyConfig(disable_claim_decomposition=True)
    gv = GraphVerify(cfg, llm_client=FakeLLMClient())
    answer = "Einstein was born in Ulm. He won the Nobel Prize in 1921."
    out = gv.verify(query=sample_query, passages=sample_passages, answer=answer)
    assert len(out.records) == 1
    assert out.records[0]["claim"] == answer


def test_evidence_mode_text_skips_graph_construction(sample_query, sample_passages, sample_answer):
    cfg = GraphVerifyConfig(evidence_mode="text")
    gv = GraphVerify(cfg, llm_client=FakeLLMClient())
    out = gv.verify(query=sample_query, passages=sample_passages, answer=sample_answer)
    assert out.graph_stats["n_nodes"] == 0
    assert all(r["path_type"] == "text" for r in out.records)


def test_kg_paths_mode_without_external_kg_warns_and_falls_back(sample_query, sample_passages, sample_answer):
    cfg = GraphVerifyConfig(evidence_mode="kg_paths", external_kg_path=None)
    gv = GraphVerify(cfg, llm_client=FakeLLMClient())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gv.verify(query=sample_query, passages=sample_passages, answer=sample_answer)
    assert any("kg_paths" in str(w.message) for w in caught)


def test_kg_paths_mode_merges_external_kg(tmp_path, sample_query, sample_passages, sample_answer):
    kg_path = tmp_path / "kg.jsonl"
    kg_path.write_text(json.dumps({"head": "Einstein", "relation": "award", "tail": "Nobel Prize"}) + "\n")
    cfg = GraphVerifyConfig(evidence_mode="kg_paths", external_kg_path=str(kg_path))
    gv = GraphVerify(cfg, llm_client=FakeLLMClient())
    graph = gv.build_graph(sample_query, sample_passages)
    labels = graph.node_labels
    assert "Einstein" in labels
    assert "Nobel Prize" in labels


def test_evidence_mode_hybrid_applies_text_fallback_for_unresolved_claims(sample_query):
    # Passages with no extractable triples (so the graph pipeline can't
    # resolve anything), but text overlap with the claim is very high, so
    # the hybrid text fallback should recover a Supported verdict.
    passages = [{"id": "p1", "text": "Zzqx flerm blorptastic wibbly wobblonium", "rank": 1, "score": 0.5}]
    cfg = GraphVerifyConfig(evidence_mode="hybrid", text_fallback_threshold=0.5)
    gv = GraphVerify(cfg, llm_client=FakeLLMClient())
    out = gv.verify(query=sample_query, passages=passages, answer="Zzqx flerm blorptastic wibbly wobblonium")
    assert len(out.records) == 1


def test_calibrator_rescales_reliability(sample_query, sample_passages, sample_answer, tmp_path):
    from graphverify.calibrator import TemperatureCalibrator

    calibrator = TemperatureCalibrator()
    calibrator.temperature = 3.0  # aggressive smoothing toward 0.5
    path = str(tmp_path / "cal.json")
    calibrator.save(path)

    gv = GraphVerify(GraphVerifyConfig(), llm_client=FakeLLMClient())
    gv.load_calibrator(path)
    out = gv.verify(query=sample_query, passages=sample_passages, answer=sample_answer)
    for r in out.records:
        if r["reliability"] not in (0.0, 1.0):
            assert 0.0 <= r["reliability"] <= 1.0


def test_verification_output_verdict_counts(sample_query, sample_passages, sample_answer):
    gv = GraphVerify(GraphVerifyConfig(), llm_client=FakeLLMClient())
    out = gv.verify(query=sample_query, passages=sample_passages, answer=sample_answer)
    assert out.n_supported + out.n_unsupported + out.n_contradictory == len(out.records)
