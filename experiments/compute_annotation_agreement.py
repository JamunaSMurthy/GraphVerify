"""
Human annotation agreement: computes Cohen's kappa and Krippendorff's alpha
for verdict labels and path-correctness labels (reported separately, as the
revision plan requires), and runs adjudication over disagreements, from
real annotation CSVs you provide.

This script performs no LLM calls and fabricates nothing: it is pure
statistics over `dataset/claim_annotation.py`-schema CSV files produced by
real human annotators following `docs/ANNOTATION_GUIDELINES.md`. Running it
without real annotation files is meaningless -- there is no synthetic
fallback.

Usage (two annotators, verdict + path-correctness agreement):
  python experiments/compute_annotation_agreement.py \\
      --annotator_files annotator1.csv,annotator2.csv \\
      --adjudicated_output output/annotations/adjudicated.csv \\
      --dataset hotpotqa
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.claim_annotation import (
    AdjudicatedRecord,
    adjudicate,
    cohens_kappa_path_correctness,
    cohens_kappa_verdicts,
    krippendorffs_alpha_path_correctness,
    krippendorffs_alpha_verdicts,
    load_annotation_csv,
)
from experiments._common import save_csv, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotator_files", type=str, required=True,
                    help="Comma-separated annotation CSVs, one per annotator (dataset/claim_annotation.py schema).")
    p.add_argument("--adjudicator_file", type=str, default=None,
                    help="Optional CSV with columns item_id,verdict giving the adjudicator's ruling for disagreements.")
    p.add_argument("--adjudicated_output", type=str, default="output/annotations/adjudicated.csv")
    p.add_argument("--report_output", type=str, default="output/annotations/agreement_report.json")
    return p.parse_args()


def load_adjudicator_rulings(path: str):
    import csv
    rulings = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rulings[row["item_id"]] = row["verdict"]
    return rulings


def main() -> None:
    args = parse_args()
    paths: List[str] = [p.strip() for p in args.annotator_files.split(",") if p.strip()]
    if len(paths) < 2:
        raise ValueError("Need at least 2 annotator files to compute inter-annotator agreement.")

    annotator_records = [load_annotation_csv(p) for p in paths]

    report = {
        "n_annotators": len(paths),
        "n_items_per_annotator": [len(r) for r in annotator_records],
        "krippendorff_alpha_verdicts": krippendorffs_alpha_verdicts(annotator_records),
        "krippendorff_alpha_path_correctness": krippendorffs_alpha_path_correctness(annotator_records),
    }
    if len(paths) == 2:
        report["cohens_kappa_verdicts"] = cohens_kappa_verdicts(annotator_records[0], annotator_records[1])
        try:
            report["cohens_kappa_path_correctness"] = cohens_kappa_path_correctness(annotator_records[0], annotator_records[1])
        except ValueError as e:
            report["cohens_kappa_path_correctness_error"] = str(e)

    print("Inter-annotator agreement:")
    for key, value in report.items():
        print(f"  {key}: {value}")

    adjudicator_rulings = load_adjudicator_rulings(args.adjudicator_file) if args.adjudicator_file else {}

    by_item = {}
    for records in annotator_records:
        for r in records:
            by_item.setdefault(r.item_id, []).append(r)

    adjudicated: List[AdjudicatedRecord] = []
    n_disagreements_unresolved = 0
    for item_id, records in sorted(by_item.items()):
        dataset = records[0].dataset
        claim = records[0].claim
        try:
            rec = adjudicate(item_id, dataset, claim, records, adjudicator_verdict=adjudicator_rulings.get(item_id))
            adjudicated.append(rec)
        except ValueError:
            n_disagreements_unresolved += 1

    report["n_items_total"] = len(by_item)
    report["n_disagreements"] = sum(1 for r in adjudicated if r.disagreement)
    report["n_disagreements_unresolved"] = n_disagreements_unresolved

    if adjudicated:
        rows = [{
            "item_id": r.item_id, "dataset": r.dataset, "claim": r.claim,
            "final_verdict": r.final_verdict, "final_path_correct": r.final_path_correct,
            "disagreement": r.disagreement, "adjudicator_id": r.adjudicator_id,
        } for r in adjudicated]
        save_csv(rows, args.adjudicated_output)
        print(f"\nSaved {len(rows)} adjudicated records to {args.adjudicated_output}")

    if n_disagreements_unresolved:
        print(f"\n[WARNING] {n_disagreements_unresolved} item(s) disagree and have no adjudicator ruling; "
              f"pass --adjudicator_file to resolve them.")

    save_json(report, args.report_output)
    print(f"Saved agreement report to {args.report_output}")


if __name__ == "__main__":
    main()
