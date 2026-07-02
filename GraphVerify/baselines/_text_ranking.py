"""Lexical passage ranking shared by FIRE and CiteFix baselines.

A dependency-free stand-in for "most likely to be retrieved/cited next"
when only a fixed, pre-retrieved passage pool is available (see the
reimplementation notes in `baselines/fire.py` and `baselines/citefix.py`).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Set

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> Set[str]:
    return set(_WORD_RE.findall(text.lower()))


def rank_passages_by_overlap(claim: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ranks passages by claim/passage token-overlap count, descending."""
    claim_tokens = tokenize(claim)
    scored = [
        (len(claim_tokens & tokenize(str(p.get("text", "")))), p)
        for p in passages
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored]
