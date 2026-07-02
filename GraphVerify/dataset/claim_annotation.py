"""
Annotation protocol tooling for claim-level and path-level human labels.

Implements the two-annotator-plus-adjudication protocol the revision plan
requires: each item is independently labeled by (at least) two annotators;
items where they agree are finalized directly, items where they disagree go
to a third adjudicator. Verdict agreement and path-correctness agreement are
computed and reported *separately* (a system can have a correct verdict
with a wrong evidence path, or vice versa, so conflating the two would hide
that failure mode).

This module does not and cannot fabricate human annotations — it is
statistics and record-keeping over annotation files you produce with real
annotators. See `docs/ANNOTATION_GUIDELINES.md` for the labeling protocol
given to annotators, and `experiments/compute_annotation_agreement.py` for
the CLI that reads annotation CSVs and reports agreement.
"""
from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Optional, Sequence

import krippendorff
import numpy as np
from sklearn.metrics import cohen_kappa_score


@dataclass
class AnnotationRecord:
    item_id:        str
    dataset:        str
    claim:          str
    annotator_id:   str
    verdict:        str                    # Supported | Unsupported | Contradictory
    path_correct:   Optional[bool] = None   # None if no path was returned to judge
    evidence_span:  str = ""
    notes:          str = ""


@dataclass
class AdjudicatedRecord:
    item_id:             str
    dataset:              str
    claim:                str
    final_verdict:        str
    final_path_correct:   Optional[bool]
    adjudicator_id:        str
    disagreement:          bool
    annotator_verdicts:    Dict[str, str]


def load_annotation_csv(path: str) -> List[AnnotationRecord]:
    """Loads annotation records from a CSV with a header matching AnnotationRecord's fields."""
    field_names = {f.name for f in fields(AnnotationRecord)}
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = field_names - {"path_correct", "evidence_span", "notes"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing required columns {sorted(missing)}")
        for row in reader:
            path_correct = row.get("path_correct", "")
            records.append(AnnotationRecord(
                item_id=row["item_id"], dataset=row["dataset"], claim=row["claim"],
                annotator_id=row["annotator_id"], verdict=row["verdict"],
                path_correct=_parse_optional_bool(path_correct),
                evidence_span=row.get("evidence_span", ""), notes=row.get("notes", ""),
            ))
    return records


def save_annotation_csv(records: Sequence[AnnotationRecord], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    field_names = [f.name for f in fields(AnnotationRecord)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


def _parse_optional_bool(raw: str) -> Optional[bool]:
    raw = (raw or "").strip().lower()
    if raw in ("", "none", "n/a"):
        return None
    return raw in ("true", "1", "yes")


def cohens_kappa_verdicts(records_a: Sequence[AnnotationRecord], records_b: Sequence[AnnotationRecord]) -> float:
    """Cohen's kappa over verdict labels between two annotators, matched by item_id."""
    labels_a, labels_b = _match_by_item(records_a, records_b, attr="verdict")
    return float(cohen_kappa_score(labels_a, labels_b))


def cohens_kappa_path_correctness(records_a: Sequence[AnnotationRecord], records_b: Sequence[AnnotationRecord]) -> float:
    """
    Cohen's kappa over path-correctness judgments between two annotators,
    matched by item_id. Items where either annotator recorded
    `path_correct=None` (no path was returned to judge) are excluded.
    """
    a_by_id = {r.item_id: r.path_correct for r in records_a if r.path_correct is not None}
    b_by_id = {r.item_id: r.path_correct for r in records_b if r.path_correct is not None}
    common = sorted(set(a_by_id) & set(b_by_id))
    if not common:
        raise ValueError("No overlapping judged (non-None) path-correctness items between the two annotators.")
    return float(cohen_kappa_score([a_by_id[i] for i in common], [b_by_id[i] for i in common]))


def _match_by_item(records_a, records_b, attr: str):
    a_by_id = {r.item_id: getattr(r, attr) for r in records_a}
    b_by_id = {r.item_id: getattr(r, attr) for r in records_b}
    common = sorted(set(a_by_id) & set(b_by_id))
    if not common:
        raise ValueError("No overlapping item_ids between the two annotators' records.")
    if set(a_by_id) != set(b_by_id):
        only_a = set(a_by_id) - set(b_by_id)
        only_b = set(b_by_id) - set(a_by_id)
        raise ValueError(
            f"Annotator item sets differ: {len(only_a)} item(s) only in A, {len(only_b)} item(s) only in B. "
            "Agreement should be computed over a fixed shared annotation batch."
        )
    return [a_by_id[i] for i in common], [b_by_id[i] for i in common]


def krippendorffs_alpha_verdicts(annotator_records: Sequence[Sequence[AnnotationRecord]]) -> float:
    """
    Krippendorff's alpha (nominal level of measurement) over verdict labels
    across two or more annotators, who need not have labeled the exact same
    item set (missing values are allowed and excluded pairwise, unlike
    Cohen's kappa which requires identical item sets).
    """
    return _krippendorff_alpha(annotator_records, attr="verdict")


def krippendorffs_alpha_path_correctness(annotator_records: Sequence[Sequence[AnnotationRecord]]) -> float:
    """As :func:`krippendorffs_alpha_verdicts`, but over path-correctness booleans (None excluded)."""
    filtered = [[r for r in records if r.path_correct is not None] for records in annotator_records]
    return _krippendorff_alpha(filtered, attr="path_correct")


def _krippendorff_alpha(annotator_records: Sequence[Sequence[AnnotationRecord]], attr: str) -> float:
    if len(annotator_records) < 2:
        raise ValueError("Krippendorff's alpha requires at least 2 annotators.")

    all_items = sorted({r.item_id for records in annotator_records for r in records})
    if not all_items:
        raise ValueError("No annotated items to compute agreement over.")
    item_index = {item: i for i, item in enumerate(all_items)}

    values = sorted({str(getattr(r, attr)) for records in annotator_records for r in records})
    value_code = {v: i for i, v in enumerate(values)}

    matrix = np.full((len(annotator_records), len(all_items)), np.nan)
    for row, records in enumerate(annotator_records):
        for r in records:
            matrix[row, item_index[r.item_id]] = value_code[str(getattr(r, attr))]

    return float(krippendorff.alpha(reliability_data=matrix, level_of_measurement="nominal"))


def adjudicate(
    item_id: str,
    dataset: str,
    claim: str,
    annotator_records: Sequence[AnnotationRecord],
    adjudicator_verdict: Optional[str] = None,
    adjudicator_id: str = "",
) -> AdjudicatedRecord:
    """
    Resolves one item's final verdict from its per-annotator records.

    If every annotator gave the same verdict, that verdict is final and
    `disagreement=False`. If they disagree, `adjudicator_verdict` must be
    supplied (raises ValueError otherwise, since a disagreement cannot be
    silently resolved) and becomes final with `disagreement=True`.

    `final_path_correct` is the majority vote over annotators who recorded
    a (non-None) path-correctness judgment, or None if nobody judged a path.
    """
    verdicts = {r.annotator_id: r.verdict for r in annotator_records}
    unique_verdicts = set(verdicts.values())

    if len(unique_verdicts) == 1:
        final_verdict = next(iter(unique_verdicts))
        disagreement = False
    else:
        if not adjudicator_verdict:
            raise ValueError(
                f"Item {item_id}: annotators disagree ({verdicts}) and no adjudicator_verdict was supplied."
            )
        final_verdict = adjudicator_verdict
        disagreement = True

    path_flags = [r.path_correct for r in annotator_records if r.path_correct is not None]
    final_path_correct = (sum(path_flags) > len(path_flags) / 2) if path_flags else None

    return AdjudicatedRecord(
        item_id=item_id, dataset=dataset, claim=claim,
        final_verdict=final_verdict, final_path_correct=final_path_correct,
        adjudicator_id=adjudicator_id, disagreement=disagreement,
        annotator_verdicts=verdicts,
    )
