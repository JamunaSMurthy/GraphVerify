"""
Determines whether a graph path tail conflicts with a claim's tail value.

Four conflict conditions:
  1. Functional relation — (head, relation) uniquely determines the tail,
     so a different graph tail is a direct contradiction.
  2. Temporal — year values differ for a date-typed relation.
  3. Numeric — values differ beyond ±5% tolerance.
  4. Mutually exclusive labels — e.g. "supports" vs "refutes".
"""
from __future__ import annotations

import re
from typing import Optional

from .config import FUNCTIONAL_RELATIONS, TEMPORAL_RELATIONS, NUMERIC_TOLERANCE, MUTUALLY_EXCLUSIVE_SETS


def is_incompatible(
    claim_tail: str,
    path_tail: str,
    relation: str,
    numeric_tolerance: float = NUMERIC_TOLERANCE,
) -> bool:
    if not claim_tail or not path_tail:
        return False

    ct_norm = _normalize_val(claim_tail)
    pt_norm = _normalize_val(path_tail)

    if ct_norm == pt_norm:
        return False

    if relation in FUNCTIONAL_RELATIONS:
        return True

    ct_year = _extract_year(claim_tail)
    pt_year = _extract_year(path_tail)
    if ct_year is not None and pt_year is not None and relation in TEMPORAL_RELATIONS:
        return ct_year != pt_year

    ct_num = _extract_number(claim_tail)
    pt_num = _extract_number(path_tail)
    if ct_num is not None and pt_num is not None:
        denom = max(abs(ct_num), abs(pt_num), 1e-9)
        if abs(ct_num - pt_num) / denom > numeric_tolerance:
            return True

    for excl_set in MUTUALLY_EXCLUSIVE_SETS:
        if ct_norm in excl_set and pt_norm in excl_set:
            return True

    return False


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
