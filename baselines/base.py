"""
Uniform interface for baseline claim-level verifiers.

Every baseline in this package converts ``(query, retrieved passages, a
pre-decomposed list of atomic claims)`` into one
:class:`graphverify.verdict_assigner.VerificationRecord` per claim, so
``experiments/run_main_verification_benchmark.py`` and every downstream
metric/analysis module (:mod:`eval.metrics`, :mod:`eval.contradiction_taxonomy`,
:mod:`eval.error_analysis`) can treat GraphVerify and all nine baselines
identically.

**Fairness protocol.** Claims are decomposed exactly once per generated
answer, by a single shared :class:`graphverify.claim_decomposer.ClaimDecomposer`,
and the *same* claim list is passed to every method under comparison
(GraphVerify-score, GraphVerify-hybrid, and all baselines below). This
matches the revision plan's baseline-fairness requirement: differences
between methods must come from verification, not from each method
decomposing the same answer differently. Callers that need this guarantee
must pass a pre-decomposed ``claims`` list to :meth:`BaselineVerifier.verify`;
the default (decomposing internally when ``claims`` is omitted) exists only
for standalone use of a single baseline outside a comparison study.

Each subclass declares:
  - ``name``: registry key used throughout ``experiments/`` and result tables.
  - ``citation``: the exact published method being approximated, plus a short
    note on how this implementation differs (we do not have access to the
    original authors' code, so every baseline here is an independent,
    from-description reimplementation used strictly as an evaluation
    control).
  - ``category``: one of ``"native_posthoc"``, ``"adapted_graph_retrieval"``,
    ``"kg_fact_checking"``, ``"ablation_control"`` — matches the
    baseline-fairness table in the revision plan so result tables can
    group/caveat rows correctly instead of presenting every baseline as an
    equally "native" post-hoc verifier. In particular,
    ``"adapted_graph_retrieval"`` baselines (GraphRAG, HippoRAG) are
    retrieval systems repurposed as verifiers by feeding their retrieved
    subgraph into GraphVerify's own shared verdict head
    (:class:`graphverify.path_searcher.PathSearcher` +
    :class:`graphverify.verdict_assigner.VerdictAssigner`) — they must never
    be reported as native verifiers without that caveat.
  - ``uses_graph``: whether the method builds/consumes any graph structure.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from graphverify.claim_decomposer import ClaimDecomposer
from graphverify.verdict_assigner import VerificationRecord


class BaselineVerifier(ABC):
    name: str = "base"
    citation: str = ""
    category: str = "native_posthoc"
    uses_graph: bool = False

    def __init__(self, llm_client: Any) -> None:
        self._llm = llm_client

    @abstractmethod
    def verify_claims(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        claims: List[str],
    ) -> List[VerificationRecord]:
        """Returns one VerificationRecord per input claim, in the same order as `claims`."""
        raise NotImplementedError

    def verify(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        answer: str,
        claims: Optional[List[str]] = None,
    ) -> List[VerificationRecord]:
        """
        Convenience entrypoint that decomposes `answer` into claims if
        `claims` is not supplied. Prefer passing a pre-decomposed `claims`
        list from a single shared decomposer when comparing multiple
        methods on the same answer — see the module docstring's fairness
        protocol.
        """
        if claims is None:
            claims = ClaimDecomposer(self._llm).decompose(answer)
        return self.verify_claims(query, passages, claims)
