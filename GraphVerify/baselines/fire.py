"""
FIRE — Fact-checking with Iterative REtrieval (Xie et al., 2025, "FIRE:
Fact-checking with Iterative Retrieval and Verification").

Original method: alternates between retrieving additional evidence and
verifying the claim against the evidence gathered so far, stopping once the
verifier is confident, and reports the evidence spans it relied on.

**Reimplementation note.** All methods in this comparison share the same
fixed, pre-retrieved passage pool (the same retrieved-evidence budget), so
there is no live retrieval to iterate over. We instead iterate over
*subsets* of that fixed pool: passages are ranked by lexical term overlap
with the claim (a simple, dependency-free proxy for "most likely to be
retrieved next"), and each round exposes the next-most-relevant batch to the
verifier, which reports whether it is confident enough to stop
(`prompts/fire_iteration_*.txt`). This preserves FIRE's defining
mechanism — verification confidence gates how much evidence is consumed —
without requiring a live retrieval index. The final round's verdict is
returned along with the evidence passage ids the model reports relying on.
"""
from __future__ import annotations

from typing import Any, Dict, List

from graphverify.prompts import load_prompt
from graphverify.verdict_assigner import VerificationRecord

from .base import BaselineVerifier
from ._response_parsing import parse_verdict_response
from ._text_ranking import rank_passages_by_overlap


class FIREVerifier(BaselineVerifier):
    name = "fire"
    citation = "Xie et al. 2025, 'FIRE: Fact-checking with Iterative Retrieval and Verification'"
    category = "native_posthoc"
    uses_graph = False

    def __init__(self, llm_client: Any, batch_size: int = 2, max_rounds: int = 4) -> None:
        super().__init__(llm_client)
        self._batch_size = batch_size
        self._max_rounds = max_rounds

    def verify_claims(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        claims: List[str],
    ) -> List[VerificationRecord]:
        records = []
        for claim in claims:
            verdict, confidence, rationale, evidence_ids = self._iterative_verify(claim, passages)
            records.append(VerificationRecord(
                claim=claim, head=None, relation="", tail=None,
                verdict=verdict, best_path=evidence_ids or None, reliability=confidence,
                support_score=confidence if verdict == "Supported" else 0.0,
                contradict_score=confidence if verdict == "Contradictory" else 0.0,
                path_type="text", triple_linked=False,
                verdict_mode=self.name, rationale=rationale,
            ))
        return records

    def _iterative_verify(self, claim: str, passages: List[Dict[str, Any]]):
        if not passages:
            return "Unsupported", 0.0, "No retrieved passages available for iterative retrieval.", []

        ranked = rank_passages_by_overlap(claim, passages)
        n_total = len(ranked)
        max_rounds = min(self._max_rounds, -(-n_total // self._batch_size))  # ceil division

        verdict, confidence, rationale, evidence_ids = "Unsupported", 0.0, "No round completed.", []
        for round_num in range(1, max_rounds + 1):
            batch = ranked[: round_num * self._batch_size]
            evidence_block = "\n".join(
                f"[{p.get('id', 'p')}] {str(p.get('text', ''))[:500]}" for p in batch
            )
            result = self._llm.chat_json([
                {"role": "system", "content": load_prompt("fire_iteration_system")},
                {"role": "user", "content": load_prompt("fire_iteration_user").format(
                    claim=claim, evidence=evidence_block,
                    n_passages=len(batch), n_total=n_total,
                    round_num=round_num, max_rounds=max_rounds,
                )},
            ])
            verdict, confidence, rationale = parse_verdict_response(result)
            evidence_ids = list(result.get("evidence_ids", [])) if isinstance(result, dict) else []

            confident_enough = bool(result.get("confident_enough", False)) if isinstance(result, dict) else False
            if confident_enough or len(batch) >= n_total:
                break

        return verdict, confidence, rationale, evidence_ids
