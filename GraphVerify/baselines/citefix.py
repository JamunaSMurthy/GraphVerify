"""
CiteFix — citation correction for RAG (Maheshwari et al., 2025, "CiteFix:
Enhancing RAG Accuracy Through Post-Processing Citation Correction").

Original method: given a generated answer with inline citations, checks
whether each cited source actually supports its statement and, if not,
searches for and substitutes a better-supporting citation ("repair").

**Reimplementation note.** The datasets used in this benchmark do not carry
explicit inline citation markers per claim, so — consistent with how
CiteFix is described as a post-processing step over whatever citation the
generator produced — the "originally cited" passage for a claim is taken to
be the passage the generator was most likely drawing on, approximated by
lexical term-overlap ranking (`baselines/_text_ranking.py`, the same
dependency-free ranking used by the FIRE baseline). CiteFix then (1) checks
whether that passage actually supports the claim
(`prompts/citefix_check_*.txt`) and (2), only if it does not, attempts
citation repair by searching the remaining candidates
(`prompts/citefix_repair_*.txt`). The verdict is Supported/Contradictory
only when a specific passage settles it (matching CiteFix's citation-level
operation); "Unsupported" covers both "cited passage insufficient and no
repair found" and "no passages at all."
"""
from __future__ import annotations

from typing import Any, Dict, List

from graphverify.prompts import load_prompt
from graphverify.verdict_assigner import VerificationRecord

from .base import BaselineVerifier
from ._response_parsing import parse_verdict_response
from ._text_ranking import rank_passages_by_overlap


class CiteFixVerifier(BaselineVerifier):
    name = "citefix"
    citation = "Maheshwari et al. 2025, 'CiteFix: Enhancing RAG Accuracy Through Post-Processing Citation Correction'"
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
            verdict, confidence, rationale, cited_id = self._check_and_repair(claim, passages)
            records.append(VerificationRecord(
                claim=claim, head=None, relation="", tail=None,
                verdict=verdict, best_path=[cited_id] if cited_id else None, reliability=confidence,
                support_score=confidence if verdict == "Supported" else 0.0,
                contradict_score=confidence if verdict == "Contradictory" else 0.0,
                path_type="text", triple_linked=False,
                verdict_mode=self.name, rationale=rationale,
            ))
        return records

    def _check_and_repair(self, claim: str, passages: List[Dict[str, Any]]):
        if not passages:
            return "Unsupported", 0.0, "No retrieved passages to cite.", None

        ranked = rank_passages_by_overlap(claim, passages)
        cited = ranked[0]
        cited_id = str(cited.get("id", "p"))

        check = self._llm.chat_json([
            {"role": "system", "content": load_prompt("citefix_check_system")},
            {"role": "user", "content": load_prompt("citefix_check_user").format(
                claim=claim, passage_id=cited_id, passage_text=str(cited.get("text", ""))[:800],
            )},
        ])
        verdict, confidence, rationale = parse_verdict_response(check)
        if verdict != "Unsupported":
            return verdict, confidence, rationale, cited_id

        remaining = ranked[1:]
        if not remaining:
            return verdict, confidence, rationale, cited_id

        evidence_block = "\n".join(
            f"[{p.get('id', 'p')}] {str(p.get('text', ''))[:500]}" for p in remaining
        )
        repair = self._llm.chat_json([
            {"role": "system", "content": load_prompt("citefix_repair_system")},
            {"role": "user", "content": load_prompt("citefix_repair_user").format(
                claim=claim, passage_id=cited_id, check_rationale=rationale, evidence=evidence_block,
            )},
        ])
        r_verdict, r_confidence, r_rationale = parse_verdict_response(repair)
        repaired_id = repair.get("repaired_passage_id") if isinstance(repair, dict) else None
        return r_verdict, r_confidence, r_rationale, (repaired_id or cited_id)
