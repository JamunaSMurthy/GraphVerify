"""
Text-only evidence check shared by two call sites:

1. ``evidence_mode="text"`` / ``"hybrid"`` in :class:`graphverify.verifier.GraphVerify`
   — for "text", it is the sole verdict source per claim; for "hybrid", it is
   a fallback applied only to claims the graph pipeline left Unsupported,
   implementing the "textual fallback matching" the method description
   names as an extension.
2. :class:`baselines.llm_text_verifier.LLMTextVerifier` — the critical
   text-only control baseline, which does nothing else.

Both call sites need the exact same behavior (claim + raw retrieved-passage
text in, three-way verdict + confidence out, no graph, no triples), so the
logic lives in one place instead of being duplicated between the core
pipeline and the baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .prompts import load_prompt
from .verdict_assigner import VERDICT_CONTRADICTORY, VERDICT_SUPPORTED, VERDICT_UNSUPPORTED

_VALID_VERDICTS = {VERDICT_SUPPORTED, VERDICT_UNSUPPORTED, VERDICT_CONTRADICTORY}


@dataclass
class TextEvidenceVerdict:
    verdict:    str
    confidence: float
    rationale:  str


def text_entailment_verdict(
    llm_client: Any,
    claim: str,
    passages: List[Dict[str, Any]],
    max_passage_chars: int = 500,
) -> TextEvidenceVerdict:
    """
    Checks a claim against raw retrieved-passage text only, with no graph or
    triple structure. Returns Unsupported with zero confidence if there are
    no passages, or if the LLM call fails / returns unparseable output — a
    text-evidence check that cannot reach a verdict is not the same as a
    contradiction, so it must never default to Contradictory.
    """
    if not passages:
        return TextEvidenceVerdict(VERDICT_UNSUPPORTED, 0.0, "No retrieved passages available.")

    evidence_block = "\n".join(
        f"[{p.get('id', 'p')}] {str(p.get('text', ''))[:max_passage_chars]}"
        for p in passages
    )
    messages = [
        {"role": "system", "content": load_prompt("text_entailment_system")},
        {"role": "user", "content": load_prompt("text_entailment_user").format(
            claim=claim, evidence=evidence_block,
        )},
    ]
    result = llm_client.chat_json(messages)
    if not isinstance(result, dict) or "verdict" not in result:
        return TextEvidenceVerdict(
            VERDICT_UNSUPPORTED, 0.0,
            "LLM entailment check returned no parseable response.",
        )

    verdict = str(result.get("verdict", "")).strip()
    if verdict not in _VALID_VERDICTS:
        verdict = VERDICT_UNSUPPORTED

    try:
        confidence = float(result.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = min(max(confidence, 0.0), 1.0)

    rationale = str(result.get("rationale", "")).strip()
    return TextEvidenceVerdict(verdict=verdict, confidence=confidence, rationale=rationale)
