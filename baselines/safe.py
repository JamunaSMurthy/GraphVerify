"""
SAFE — Search-Augmented Factuality Evaluator (Wei et al., 2024,
"Long-form factuality in large language models").

Original method: decompose a long-form response into individual atomic
facts, then for each fact issue search-engine queries and rate the fact
Supported / Irrelevant / Not-Supported against the retrieved search results.

**Reimplementation note.** This repository has no live web-search tool, so
the fixed set of retrieved passages already available to every method in
the comparison (the same retrieved-evidence budget used by GraphVerify)
stands in for SAFE's search results — this is the standard substitution
used across this baseline set (see `baselines/base.py` module docstring) and
keeps the comparison apples-to-apples: every method sees the same evidence.
SAFE natively has no explicit contradiction category; we surface an explicit
conflicting statement as "Contradictory" only when the LLM finds one,
rather than silently merging it into "Unsupported" (see `prompts/safe_system.txt`).
Claim decomposition is not performed here — it is shared across all methods
per the fairness protocol in `baselines/base.py`.
"""
from __future__ import annotations

from typing import Any, Dict, List

from graphverify.prompts import load_prompt
from graphverify.verdict_assigner import VerificationRecord

from .base import BaselineVerifier
from ._response_parsing import parse_verdict_response


class SAFEVerifier(BaselineVerifier):
    name = "safe"
    citation = "Wei et al. 2024, 'Long-form factuality in large language models' (SAFE)"
    category = "native_posthoc"
    uses_graph = False

    def verify_claims(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        claims: List[str],
    ) -> List[VerificationRecord]:
        records = []
        for claim in claims:
            verdict, confidence, rationale = self._rate_fact(claim, passages)
            records.append(VerificationRecord(
                claim=claim, head=None, relation="", tail=None,
                verdict=verdict, best_path=None, reliability=confidence,
                support_score=confidence if verdict == "Supported" else 0.0,
                contradict_score=confidence if verdict == "Contradictory" else 0.0,
                path_type="text", triple_linked=False,
                verdict_mode=self.name, rationale=rationale,
            ))
        return records

    def _rate_fact(self, claim: str, passages: List[Dict[str, Any]]):
        if not passages:
            return "Unsupported", 0.0, "No search results (retrieved passages) available."
        evidence_block = "\n".join(
            f"[{p.get('id', 'p')}] {str(p.get('text', ''))[:500]}" for p in passages
        )
        messages = [
            {"role": "system", "content": load_prompt("safe_system")},
            {"role": "user", "content": load_prompt("safe_user").format(claim=claim, evidence=evidence_block)},
        ]
        result = self._llm.chat_json(messages)
        return parse_verdict_response(result)
