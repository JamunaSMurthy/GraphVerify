"""Shared JSON-verdict response parsing used by several baseline modules."""
from __future__ import annotations

from typing import Any, Tuple

from graphverify.verdict_assigner import VERDICT_CONTRADICTORY, VERDICT_SUPPORTED, VERDICT_UNSUPPORTED

VALID_VERDICTS = {VERDICT_SUPPORTED, VERDICT_UNSUPPORTED, VERDICT_CONTRADICTORY}


def parse_verdict_response(result: Any, default_confidence: float = 0.5) -> Tuple[str, float, str]:
    """
    Parses a ``{"verdict", "confidence", "rationale"}`` LLM response into a
    validated ``(verdict, confidence, rationale)`` tuple. Falls back to
    Unsupported/0.0 confidence on any parse failure so an unparseable
    response never silently becomes a false Contradictory or Supported
    verdict.
    """
    if not isinstance(result, dict) or "verdict" not in result:
        return VERDICT_UNSUPPORTED, 0.0, "LLM call returned no parseable response."

    verdict = str(result.get("verdict", "")).strip()
    if verdict not in VALID_VERDICTS:
        verdict = VERDICT_UNSUPPORTED

    try:
        confidence = float(result.get("confidence", default_confidence))
    except (TypeError, ValueError):
        confidence = default_confidence
    confidence = min(max(confidence, 0.0), 1.0)

    rationale = str(result.get("rationale", "")).strip()
    return verdict, confidence, rationale
