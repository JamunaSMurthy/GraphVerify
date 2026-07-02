"""
Unified method registry for experiment scripts.

Maps a method name to a constructor producing an object with exactly one
method, ``verify(query, passages, claims) -> List[Dict]``, regardless of
whether the underlying method is GraphVerify itself
(:mod:`graphverify.verifier`) or a `baselines/` entry
(:class:`baselines.base.BaselineVerifier`). Every experiment script that
compares methods should go through :func:`build_method` rather than
constructing `GraphVerify`/`HybridGraphVerify`/a baseline class directly, so
adding a new method to `baselines/` automatically makes it available to
every experiment script with no other code changes.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Protocol

from baselines import BASELINE_REGISTRY
from graphverify.config import GraphVerifyConfig
from graphverify.llm_client import LLMClient
from graphverify.verdict_assigner import record_to_dict
from graphverify.verifier import GraphVerify, HybridGraphVerify

GRAPHVERIFY_METHOD_NAMES = ("graphverify_score", "graphverify_hybrid")
ALL_METHOD_NAMES = GRAPHVERIFY_METHOD_NAMES + tuple(BASELINE_REGISTRY.keys())


class Method(Protocol):
    def verify(self, query: str, passages: List[Dict[str, Any]], claims: List[str]) -> List[Dict[str, Any]]:
        ...


class _GraphVerifyAdapter:
    def __init__(self, verifier) -> None:
        self._verifier = verifier

    def verify(self, query: str, passages: List[Dict[str, Any]], claims: List[str]) -> List[Dict[str, Any]]:
        output = self._verifier.verify(query=query, passages=passages, answer="", claims=claims)
        return output.records  # already plain dicts (VerificationOutput serializes via record_to_dict)


class _BaselineAdapter:
    def __init__(self, baseline) -> None:
        self._baseline = baseline

    def verify(self, query: str, passages: List[Dict[str, Any]], claims: List[str]) -> List[Dict[str, Any]]:
        records = self._baseline.verify_claims(query, passages, claims)
        return [record_to_dict(r) for r in records]


def build_method(name: str, llm_client: LLMClient, cfg: Optional[GraphVerifyConfig] = None) -> Method:
    """
    Constructs a uniform verifier for `name` (one of `ALL_METHOD_NAMES`).

    `cfg` provides shared settings (embedding model, thresholds, graph
    search hyperparameters, evidence_mode, etc.) for every method that uses
    them; `graphverify_score`/`graphverify_hybrid` each override only
    `verdict_mode` on a copy of `cfg` so the rest of the configuration is
    identical between the two GraphVerify variants. Raises ValueError for
    an unrecognized method name.
    """
    cfg = cfg or GraphVerifyConfig()

    if name == "graphverify_score":
        return _GraphVerifyAdapter(GraphVerify(replace(cfg, verdict_mode="score_only"), llm_client=llm_client))
    if name == "graphverify_hybrid":
        return _GraphVerifyAdapter(HybridGraphVerify(replace(cfg, verdict_mode="hybrid_llm"), llm_client=llm_client))
    if name in BASELINE_REGISTRY:
        return _BaselineAdapter(BASELINE_REGISTRY[name](llm_client))

    raise ValueError(f"Unknown method '{name}'. Choose from {ALL_METHOD_NAMES}")
