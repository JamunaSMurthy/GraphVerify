"""
Determines whether a graph path tail conflicts with a claim's tail value.

Four conflict conditions, checked in priority order and exposed together by
:func:`classify_incompatibility` so downstream error/contradiction-taxonomy
analysis (``eval/contradiction_taxonomy.py``) can report *which* rule fired
without re-implementing the detection logic:

  1. ``entity_functional`` — (head, relation) uniquely determines the tail,
     so a different graph tail is a direct contradiction (e.g. a person can
     only have one ``birthPlace``).
  2. ``temporal`` — year values differ for a date-typed relation.
  3. ``numeric`` — values differ beyond the configured tolerance.
  4. ``mutually_exclusive`` — e.g. "supports" vs "refutes" for the same
     subject; also covers the general case where entity/relation string
     agreement is high but the tail values are simply unequal and none of
     the more specific rules apply.
"""
from __future__ import annotations

import re
from typing import Optional

from .config import FUNCTIONAL_RELATIONS, TEMPORAL_RELATIONS, NUMERIC_TOLERANCE, MUTUALLY_EXCLUSIVE_SETS


def classify_incompatibility(
    claim_tail: str,
    path_tail: str,
    relation: str,
    numeric_tolerance: float = NUMERIC_TOLERANCE,
) -> Optional[str]:
    """
    Returns the conflict category name if `claim_tail` and `path_tail` are
    incompatible under `relation`, or None if they are compatible (equal, or
    no rule detects a conflict).

    Category names: "entity_functional", "temporal", "numeric",
    "mutually_exclusive". A caller that only needs a boolean should use
    :func:`is_incompatible`, which is exactly
    ``classify_incompatibility(...) is not None``.
    """
    if not claim_tail or not path_tail:
        return None

    ct_norm = _normalize_val(claim_tail)
    pt_norm = _normalize_val(path_tail)

    if ct_norm == pt_norm:
        return None

    if relation in FUNCTIONAL_RELATIONS:
        return "entity_functional"

    ct_year = _extract_year(claim_tail)
    pt_year = _extract_year(path_tail)
    if ct_year is not None and pt_year is not None and relation in TEMPORAL_RELATIONS:
        return "temporal" if ct_year != pt_year else None

    ct_num = _extract_number(claim_tail)
    pt_num = _extract_number(path_tail)
    if ct_num is not None and pt_num is not None:
        denom = max(abs(ct_num), abs(pt_num), 1e-9)
        if abs(ct_num - pt_num) / denom > numeric_tolerance:
            return "numeric"

    for excl_set in MUTUALLY_EXCLUSIVE_SETS:
        if ct_norm in excl_set and pt_norm in excl_set:
            return "mutually_exclusive"

    return None


def is_incompatible(
    claim_tail: str,
    path_tail: str,
    relation: str,
    numeric_tolerance: float = NUMERIC_TOLERANCE,
) -> bool:
    """Boolean conflict check. See :func:`classify_incompatibility` for the category."""
    return classify_incompatibility(claim_tail, path_tail, relation, numeric_tolerance) is not None


def _normalize_val(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def _extract_year(s: str) -> Optional[int]:
    m = re.search(r'\b(1[0-9]{3}|2[0-9]{3})\b', s)
    return int(m.group(1)) if m else None


def _extract_number(s: str) -> Optional[float]:
    s = re.sub(r',', '', s)
    m = re.search(r'([\d]+\.[\d]*|[\d]+)', s)
    if m:
        val = float(m.group(1))
        sl = s.lower()
        if "million" in sl:
            val *= 1e6
        elif "billion" in sl:
            val *= 1e9
        elif "thousand" in sl:
            val *= 1e3
        return val
    return None
