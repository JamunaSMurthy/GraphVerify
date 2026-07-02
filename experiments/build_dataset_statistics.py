"""
Builds the dataset/claim statistics table: per-dataset counts of
questions/answers, generated answers, extracted claims, verdict label
distribution, manually-labeled items, and path labels, plus a Total row.

Every number here is computed from files that actually exist. This script
never invents a claim or label count for a pipeline stage you have not run
yet — the claim/verdict columns are 0 unless you point `--predictions_dir`
at real `verify.py` output, and `manual_labels` is 0 unless you point
`--annotations_dir` at real annotation CSVs
(`dataset/claim_annotation.py` schema).

Usage:
  python experiments/build_dataset_statistics.py \\
      --datasets hotpotqa,2wikimultihopqa,musique,fever,ragtruth \\
      --split validation \\
      --predictions_dir output/predictions \\
      --annotations_dir output/annotations \\
      --output output/results/dataset_statistics.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.claim_annotation import load_annotation_csv
from experiments._common import add_dataset_args, load_samples, parse_dataset_list, save_csv

_VERDICT_KEY = {"Supported": "supported", "Unsupported": "unsupported", "Contradictory": "contradictory"}
_STAT_FIELDS = [
    "dataset", "questions_answers", "generated_answers", "extracted_claims",
    "supported", "unsupported", "contradictory", "manual_labels", "path_labels",
]


def compute_dataset_statistics(
    dataset: str,
    samples: List[Dict[str, Any]],
    prediction_files: Optional[List[str]] = None,
    annotation_files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "dataset": dataset,
        "questions_answers": len(samples),
        "generated_answers": sum(1 for s in samples if s.get("generated")),
        "path_labels": sum(1 for s in samples if s.get("gold_path")),
    }

    claim_counts = _claim_counts_from_predictions(prediction_files or [])
    stats.update(claim_counts or {"extracted_claims": 0, "supported": 0, "unsupported": 0, "contradictory": 0})
    stats["manual_labels"] = _manual_label_count(annotation_files or [], dataset)
    return stats


def _claim_counts_from_predictions(prediction_files: List[str]) -> Optional[Dict[str, int]]:
    if not prediction_files:
        return None
    counts = {"extracted_claims": 0, "supported": 0, "unsupported": 0, "contradictory": 0}
    for path in prediction_files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                for claim_rec in rec.get("records", []):
                    counts["extracted_claims"] += 1
                    key = _VERDICT_KEY.get(claim_rec.get("verdict"))
                    if key:
                        counts[key] += 1
    return counts


def _manual_label_count(annotation_files: List[str], dataset: str) -> int:
    total = 0
    for path in annotation_files:
        records = load_annotation_csv(path)
        total += sum(1 for r in records if r.dataset == dataset)
    return total


def _total_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total: Dict[str, Any] = {"dataset": "Total"}
    for key in _STAT_FIELDS[1:]:
        total[key] = sum(r.get(key, 0) for r in rows)
    return total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(p)
    p.add_argument("--predictions_dir", type=str, default=None,
                    help="Directory of verify.py output (used to count extracted claims and per-claim verdicts, if present).")
    p.add_argument("--annotations_dir", type=str, default=None,
                    help="Directory of annotation CSVs to count manually-labeled items.")
    p.add_argument("--output", type=str, default="output/results/dataset_statistics.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    datasets = parse_dataset_list(args.datasets)
    annotation_files = sorted(glob.glob(os.path.join(args.annotations_dir, "*.csv"))) if args.annotations_dir else []

    rows = []
    for dataset in datasets:
        samples = load_samples(dataset, args)
        prediction_files = (
            sorted(glob.glob(os.path.join(args.predictions_dir, dataset, "*.jsonl")))
            if args.predictions_dir else []
        )
        rows.append(compute_dataset_statistics(dataset, samples, prediction_files, annotation_files))
    rows.append(_total_row(rows))

    header = f"{'Dataset':<18} {'Q/A':>7} {'Gen':>7} {'Claims':>8} {'Supp':>6} {'Unsup':>6} {'Contr':>6} {'Manual':>7} {'Paths':>7}"
    print(header)
    for r in rows:
        print(f"{r['dataset']:<18} {r.get('questions_answers', 0):>7} {r.get('generated_answers', 0):>7} "
              f"{r.get('extracted_claims', 0):>8} {r.get('supported', 0):>6} {r.get('unsupported', 0):>6} "
              f"{r.get('contradictory', 0):>6} {r.get('manual_labels', 0):>7} {r.get('path_labels', 0):>7}")

    save_csv(rows, args.output, fieldnames=_STAT_FIELDS)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
