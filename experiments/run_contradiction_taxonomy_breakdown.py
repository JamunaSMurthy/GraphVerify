"""
Contradiction-type breakdown: classifies every Contradictory verdict a
method produced into the taxonomy from `eval/contradiction_taxonomy.py`
(entity, relation, numeric, temporal, multi-hop, mutually-exclusive-label),
turning an aggregate contradiction F1 into something a reviewer can audit
by mechanism rather than a single opaque number.

Reads prediction files produced by `experiments/run_main_verification_benchmark.py`
(``<output_dir>/<dataset>/<method>_predictions.json``); does not re-run
verification.

Usage:
  python experiments/run_contradiction_taxonomy_breakdown.py \\
      --predictions_dir output/results/main_benchmark \\
      --datasets hotpotqa,fever \\
      --methods graphverify_hybrid,graphverify_score \\
      --output output/results/contradiction_taxonomy.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.contradiction_taxonomy import TAXONOMY_CATEGORIES, contradiction_taxonomy_breakdown
from experiments._common import parse_dataset_list, save_csv


def load_predictions(predictions_dir: str, dataset: str, method: str) -> List[Dict[str, Any]]:
    path = os.path.join(predictions_dir, dataset, f"{method}_predictions.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        samples = json.load(f)
    return [rec for sample in samples for rec in sample.get("records", [])]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions_dir", type=str, required=True)
    p.add_argument("--datasets", type=str, required=True)
    p.add_argument("--methods", type=str, required=True)
    p.add_argument("--output", type=str, default="output/results/contradiction_taxonomy.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    datasets = parse_dataset_list(args.datasets)
    methods = parse_dataset_list(args.methods)

    rows = []
    print(f"{'Dataset':<16} {'Method':<20} " + " ".join(f"{c[:12]:>13}" for c in TAXONOMY_CATEGORIES))
    for dataset in datasets:
        for method in methods:
            records = load_predictions(args.predictions_dir, dataset, method)
            breakdown = contradiction_taxonomy_breakdown(records)
            row = {"dataset": dataset, "method": method}
            row.update({cat: breakdown.get(cat, 0) for cat in TAXONOMY_CATEGORIES})
            rows.append(row)
            print(f"{dataset:<16} {method:<20} " + " ".join(f"{row[c]:>13}" for c in TAXONOMY_CATEGORIES))

    save_csv(rows, args.output, fieldnames=["dataset", "method"] + list(TAXONOMY_CATEGORIES))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
