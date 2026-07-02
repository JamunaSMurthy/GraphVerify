"""
LLM-text-only three-way verifier — the critical control baseline.

Per the revision plan's baseline-fairness table, this is not an
approximation of a specific published system; it is the control every graph
method must beat: each claim plus the raw retrieved-passage text (no graph,
no triples, no path search) is judged directly by an LLM via
:func:`graphverify.text_evidence.text_entailment_verdict`. Comparing
GraphVerify against this baseline isolates what the provenance-linked
evidence graph adds beyond an LLM judging text evidence directly, on
identical claims and identical retrieved passages.
"""
from __future__ import annotations

from typing import Any, Dict, List

from graphverify.text_evidence import text_entailment_verdict
from graphverify.verdict_assigner import VerificationRecord

from .base import BaselineVerifier


class LLMTextVerifier(BaselineVerifier):
    name = "llm_text_verifier"
    citation = (
        "Critical control baseline (no single published source): a claim + "
        "raw retrieved-text verifier with no graph structure, used to "
        "isolate the contribution of GraphVerify's evidence graph."
    )
    category = "ablation_control"
    uses_graph = False

    def verify_claims(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        claims: List[str],
    ) -> List[VerificationRecord]:
        records = []
        for claim in claims:
            result = text_entailment_verdict(self._llm, claim, passages)
            records.append(VerificationRecord(
                claim=claim, head=None, relation="", tail=None,
                verdict=result.verdict, best_path=None, reliability=result.confidence,
                support_score=result.confidence if result.verdict == "Supported" else 0.0,
                contradict_score=result.confidence if result.verdict == "Contradictory" else 0.0,
                path_type="text", triple_linked=False,
                verdict_mode=self.name, rationale=result.rationale,
            ))
        return records
