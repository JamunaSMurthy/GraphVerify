"""
RARR — Research and Revise (Gao et al., 2023a, "RARR: Researching and
Revising What Language Models Say, Using Language Models").

Original method: for each statement, generate verification questions,
research answers via retrieval, and revise the statement to agree with the
evidence found ("Research" + "Editor" steps); the edit distance/agreement
between the original and revised statement is used for attribution.

**Reimplementation note.** We use RARR's verification-question mechanism as
the read-out for a claim-level verdict rather than performing a full text
revision (there is no "generated answer" to rewrite in place here — the
downstream object of this benchmark is a claim-level verdict, matching how
the revision plan's baseline table treats RARR: "No" for claims/path input,
used purely for its attribution-agreement signal). Two real LLM steps are
run per claim: (1) generate a verification question and answer it strictly
from the retrieved evidence (`prompts/rarr_question_*.txt`), (2) decide
whether that evidence-grounded answer agrees, disagrees, or fails to settle
the original claim (`prompts/rarr_agreement_*.txt`) — this is RARR's
Research + Editor agreement check, adapted to produce a three-way verdict.
"""
from __future__ import annotations

from typing import Any, Dict, List

from graphverify.prompts import load_prompt
from graphverify.verdict_assigner import VerificationRecord

from .base import BaselineVerifier
from ._response_parsing import parse_verdict_response


class RARRVerifier(BaselineVerifier):
    name = "rarr"
    citation = "Gao et al. 2023a, 'RARR: Researching and Revising What Language Models Say, Using Language Models'"
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
            verdict, confidence, rationale = self._research_and_check(claim, passages)
            records.append(VerificationRecord(
                claim=claim, head=None, relation="", tail=None,
                verdict=verdict, best_path=None, reliability=confidence,
                support_score=confidence if verdict == "Supported" else 0.0,
                contradict_score=confidence if verdict == "Contradictory" else 0.0,
                path_type="text", triple_linked=False,
                verdict_mode=self.name, rationale=rationale,
            ))
        return records

    def _research_and_check(self, claim: str, passages: List[Dict[str, Any]]):
        if not passages:
            return "Unsupported", 0.0, "No retrieved evidence to research the claim against."

        evidence_block = "\n".join(
            f"[{p.get('id', 'p')}] {str(p.get('text', ''))[:500]}" for p in passages
        )

        research = self._llm.chat_json([
            {"role": "system", "content": load_prompt("rarr_question_system")},
            {"role": "user", "content": load_prompt("rarr_question_user").format(
                claim=claim, evidence=evidence_block,
            )},
        ])
        if not isinstance(research, dict) or not research.get("evidence_found", False):
            return "Unsupported", 0.0, "RARR research step found no evidence answering the verification question."

        agreement = self._llm.chat_json([
            {"role": "system", "content": load_prompt("rarr_agreement_system")},
            {"role": "user", "content": load_prompt("rarr_agreement_user").format(
                claim=claim,
                question=research.get("question", ""),
                answer_from_evidence=research.get("answer_from_evidence", ""),
            )},
        ])
        return parse_verdict_response(agreement)
