"""
Classifies each Contradictory verdict into a taxonomy: entity, relation,
numeric, temporal, multi-hop, or mutually-exclusive-label conflict.

Reuses :func:`graphverify.incompatibility.classify_incompatibility` for the
value-conflict rules rather than re-implementing them, and adds two
structural checks that function does not cover (it only looks at the tail
value, not the relation string or path length):

  - ``relation_mismatch``: the conflicting path's relation differs from the
    claim's canonical relation even though the entities matched — the
    graph found *something* connecting the same two entities, but not
    under the relation the claim asserts.
  - ``multi_hop``: the conflict was only detectable via a path of more than
    one edge (no single retrieved edge alone states the conflicting value).

Turning contradiction F1 into something a reviewer can trust requires
knowing *why* claims were flagged as contradictory, not just how many were
— this module is what makes that auditable.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from graphverify.incompatibility import classify_incompatibility

TAXONOMY_CATEGORIES = (
    "entity_functional", "relation_mismatch", "temporal",
    "numeric", "mutually_exclusive", "multi_hop", "unknown",
)


def classify_contradiction(record: Dict[str, Any]) -> Optional[str]:
    """
    Classifies one predicted claim record's contradiction into a taxonomy
    category. Returns None if the record is not a Contradictory verdict, or
    if it carries no graph path to classify (e.g. a text-fallback or
    LLM-text-only contradiction with no ``best_path`` edges) — this
    taxonomy is specifically about graph-structural conflict reasons, so a
    non-graph contradiction is out of its scope rather than force-fit into
    "unknown".
    """
    if record.get("verdict") != "Contradictory":
        return None

    path = record.get("best_path")
    if not path or not isinstance(path, list) or not all(isinstance(e, dict) for e in path):
        return None

    last_edge = path[-1]
    claim_tail = record.get("tail") or ""
    claim_relation = record.get("relation", "")
    path_tail = last_edge.get("dst_label", last_edge.get("dst", ""))
    path_relation = last_edge.get("relation", "")

    if claim_relation and path_relation and claim_relation != path_relation:
        return "relation_mismatch"

    value_conflict = classify_incompatibility(claim_tail, path_tail, claim_relation)
    if value_conflict is not None:
        return value_conflict

    if len(path) > 1:
        return "multi_hop"

    return "unknown"


def contradiction_taxonomy_breakdown(records: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Counts contradiction records by taxonomy category across a prediction
    set. Records that are not Contradictory, or whose contradiction has no
    classifiable graph path, are excluded from the counts entirely (not
    folded into "unknown"), so the denominator reported alongside this
    breakdown should be "classifiable contradictions" — i.e.
    ``sum(counts.values())`` — not the total number of predictions.
    """
    counts: Counter = Counter()
    for rec in records:
        category = classify_contradiction(rec)
        if category is not None:
            counts[category] += 1
    return dict(counts)
