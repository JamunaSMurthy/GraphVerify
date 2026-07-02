"""
Samples verifier errors and buckets them toward the seven-category error
taxonomy the revision plan requires (claim decomposition, triple
canonicalization, entity linking, missing retrieval evidence, false
conflict, temporal/numeric failure, LLM verdict error).

This module does **not** claim to fully automate root-cause labeling — two
categories (claim decomposition error, triple canonicalization error)
fundamentally require a human to read the original generated answer against
the extracted claim/triple, which isn't reconstructable from a flattened
prediction record alone. What it does automatically is: (1) sample a
deterministic, reproducible batch of mispredicted claims, and (2) attach a
heuristic `suggested_category` from signals that *are* available in a
prediction record (whether the triple linked, the returned path type,
verdict-mode, and simple lexical cues for temporal/numeric claims). The
output CSV has a separate `human_category`/`notes` pair of columns for a
human auditor to confirm or correct the suggestion — this is the "at least
100 sampled errors categorized" deliverable, produced as an auditable
artifact, not a fabricated automatic verdict on root cause.
"""
from __future__ import annotations

import csv
import os
import random
from collections import Counter
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List

ERROR_CATEGORIES = (
    "claim_decomposition_error",
    "triple_canonicalization_error",
    "entity_linking_error",
    "missing_retrieval_evidence",
    "false_conflict",
    "temporal_numeric_failure",
    "llm_verdict_error",
    "other",
)

_TEMPORAL_NUMERIC_HINTS = ("%", "percent", "million", "billion", "thousand", "year", "date")


@dataclass
class ErrorSample:
    item_id:             str
    dataset:              str
    claim:                str
    predicted_verdict:    str
    gold_verdict:         str
    triple_linked:        bool
    path_type:            str
    reliability:          float
    verdict_mode:          str
    suggested_category:    str
    human_category:        str = ""
    notes:                 str = ""


def sample_errors(
    records: List[Dict[str, Any]],
    dataset: str,
    n: int = 100,
    seed: int = 0,
) -> List[ErrorSample]:
    """
    Deterministically samples up to `n` mispredicted claim records (where
    the predicted verdict differs from `gold_verdict`) and attaches a
    heuristic category suggestion to each. `records` must carry ``"id"``
    (item/claim id), ``"claim"``, ``"verdict"``, ``"gold_verdict"``, and the
    fields :func:`_suggest_category` reads (``"triple_linked"``,
    ``"path_type"``, ``"reliability"``, ``"verdict_mode"``).
    """
    errors = [r for r in records if r.get("verdict") != r.get("gold_verdict", "")]
    rng = random.Random(seed)
    sampled = rng.sample(errors, min(n, len(errors)))
    return [_to_error_sample(r, dataset) for r in sampled]


def _to_error_sample(record: Dict[str, Any], dataset: str) -> ErrorSample:
    return ErrorSample(
        item_id=str(record.get("id", "")),
        dataset=dataset,
        claim=str(record.get("claim", "")),
        predicted_verdict=str(record.get("verdict", "")),
        gold_verdict=str(record.get("gold_verdict", "")),
        triple_linked=bool(record.get("triple_linked", False)),
        path_type=str(record.get("path_type", "")),
        reliability=float(record.get("reliability", 0.0)),
        verdict_mode=str(record.get("verdict_mode", "")),
        suggested_category=_suggest_category(record),
    )


def _suggest_category(record: Dict[str, Any]) -> str:
    if not record.get("triple_linked", True):
        return "entity_linking_error"

    pred = record.get("verdict", "")
    gold = record.get("gold_verdict", "")
    path_type = record.get("path_type", "")

    if pred == "Unsupported" and gold in ("Supported", "Contradictory") and path_type == "none":
        return "missing_retrieval_evidence"
    if pred == "Contradictory" and gold != "Contradictory":
        return "false_conflict"

    claim_text = str(record.get("claim", "")).lower()
    if any(hint in claim_text for hint in _TEMPORAL_NUMERIC_HINTS):
        return "temporal_numeric_failure"

    if record.get("verdict_mode") == "hybrid_llm":
        return "llm_verdict_error"

    return "other"


def save_error_samples_csv(samples: List[ErrorSample], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    field_names = [f.name for f in fields(ErrorSample)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for s in samples:
            writer.writerow(asdict(s))


def load_error_samples_csv(path: str) -> List[ErrorSample]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            ErrorSample(
                item_id=row["item_id"], dataset=row["dataset"], claim=row["claim"],
                predicted_verdict=row["predicted_verdict"], gold_verdict=row["gold_verdict"],
                triple_linked=row["triple_linked"].strip().lower() == "true",
                path_type=row["path_type"], reliability=float(row["reliability"] or 0.0),
                verdict_mode=row["verdict_mode"], suggested_category=row["suggested_category"],
                human_category=row.get("human_category", ""), notes=row.get("notes", ""),
            )
            for row in reader
        ]


def summarize_error_categories(samples: List[ErrorSample]) -> Dict[str, int]:
    """Counts samples by `human_category` when set, falling back to `suggested_category` otherwise."""
    counts: Counter = Counter()
    for s in samples:
        counts[s.human_category or s.suggested_category] += 1
    return dict(counts)
