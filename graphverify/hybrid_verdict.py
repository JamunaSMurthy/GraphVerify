"""
GraphVerify-Hybrid verdict head (``verdict_mode="hybrid_llm"``).

The revision plan's central method-identity fix is to stop letting Figure-2-
style "graph-enhanced LLM verifier" language and the threshold equations
describe two different systems, and instead report two named, separately
evaluated variants:

  - GraphVerify-score (``verdict_mode="score_only"``,
    :class:`graphverify.verifier.GraphVerify`): the rule-based path scorer
    and threshold verdict assignment
    (:mod:`graphverify.path_scorer`, :mod:`graphverify.verdict_assigner`) is
    the entire decision procedure. Fully auditable, no LLM in the verdict
    loop.
  - GraphVerify-hybrid (``verdict_mode="hybrid_llm"``,
    :class:`graphverify.verifier.HybridGraphVerify`): the same rule-based
    pipeline runs first and produces a prior verdict plus the top scored
    support/conflict paths, then this module's :class:`HybridVerdictHead`
    reads the claim, canonical triple, those same paths, and the prior, and
    outputs the final verdict and a confidence in [0, 1].

The LLM verdict head is given nothing beyond what the rule-based path search
already retrieved from the shared evidence graph — no additional passages,
no gold labels, no privileged information — so GraphVerify-score and
GraphVerify-hybrid are comparable on identical evidence, isolating the
effect of letting an LLM interpret the retrieved paths from the effect of
retrieving more evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .path_scorer import ScoredPath
from .prompts import load_prompt
from .verdict_assigner import (
    VERDICT_CONTRADICTORY,
    VERDICT_SUPPORTED,
    VERDICT_UNSUPPORTED,
    VerificationRecord,
)

_VALID_VERDICTS = {VERDICT_SUPPORTED, VERDICT_UNSUPPORTED, VERDICT_CONTRADICTORY}


@dataclass
class HybridVerdict:
    verdict:             str
    confidence:          float
    rationale:           str
    overrode_rule_based: bool


class HybridVerdictHead:
    """LLM verdict head consumed by :class:`graphverify.verifier.HybridGraphVerify`."""

    def __init__(self, llm_client) -> None:
        self._llm = llm_client

    def decide(
        self,
        claim: str,
        head: Optional[str],
        relation: str,
        tail: Optional[str],
        rule_based: VerificationRecord,
        support_paths: List[ScoredPath],
        conflict_paths: List[ScoredPath],
    ) -> HybridVerdict:
        """
        Produces the final hybrid verdict for one claim.

        `rule_based` is the score-only :class:`~graphverify.verdict_assigner.VerificationRecord`
        for the same claim, included in the prompt as a prior (not as ground
        truth). If the claim's triple never linked to the graph at all,
        there is no path evidence for the LLM to reason over, so the
        rule-based Unsupported verdict is returned unchanged without an LLM
        call.
        """
        if not rule_based.triple_linked:
            return HybridVerdict(
                verdict=VERDICT_UNSUPPORTED,
                confidence=0.0,
                rationale="Claim entities did not link to the evidence graph; no paths to evaluate.",
                overrode_rule_based=False,
            )

        messages = [
            {"role": "system", "content": load_prompt("hybrid_verdict_system")},
            {"role": "user", "content": load_prompt("hybrid_verdict_user").format(
                claim=claim,
                head=head or "?", relation=relation or "?", tail=tail or "?",
                support_paths=_format_paths(support_paths[:3]) or "(none found)",
                conflict_paths=_format_paths(conflict_paths[:3]) or "(none found)",
                rule_based_verdict=rule_based.verdict,
                rule_based_score=f"{rule_based.reliability:.3f}",
            )},
        ]
        result = self._llm.chat_json(messages)
        return self._parse(result, rule_based)

    def _parse(self, result, rule_based: VerificationRecord) -> HybridVerdict:
        if not isinstance(result, dict) or "verdict" not in result:
            # The LLM call failed or returned unparseable output: fall back
            # to the auditable rule-based verdict rather than guessing.
            return HybridVerdict(
                verdict=rule_based.verdict,
                confidence=rule_based.reliability,
                rationale="LLM verdict head returned no parseable response; used the rule-based prior.",
                overrode_rule_based=False,
            )

        verdict = str(result.get("verdict", "")).strip()
        if verdict not in _VALID_VERDICTS:
            verdict = rule_based.verdict

        try:
            confidence = float(result.get("confidence", rule_based.reliability))
        except (TypeError, ValueError):
            confidence = rule_based.reliability
        confidence = min(max(confidence, 0.0), 1.0)

        rationale = str(result.get("rationale", "")).strip()
        return HybridVerdict(
            verdict=verdict,
            confidence=confidence,
            rationale=rationale,
            overrode_rule_based=(verdict != rule_based.verdict),
        )


def _format_paths(paths: List[ScoredPath]) -> str:
    lines = []
    for p in paths:
        chain = " -> ".join(
            f"{e.get('src_label', e.get('src', ''))} --[{e.get('relation', '')}]--> "
            f"{e.get('dst_label', e.get('dst', ''))}"
            for e in p.path_edges
        )
        lines.append(f"- (path score={p.score:.2f}) {chain}")
    return "\n".join(lines)
