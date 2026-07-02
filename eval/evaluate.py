"""
Multi-dataset evaluation driver.

Runs evaluation over all benchmark datasets and prints a results table
with mean ± std across verifier seeds.

Usage:
  python eval/evaluate.py \\
      --pred_root output/predictions \\
      --datasets hotpotqa,2wikimultihopqa,musique,fever,ragtruth \\
      --seeds 0,1,2 \\
      --output output/results/summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from evaluate import evaluate_seed


DATASETS = ["hotpotqa", "2wikimultihopqa", "musique", "fever", "ragtruth"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_root",  type=str, required=True)
    p.add_argument("--datasets",   type=str, default=",".join(DATASETS))
    p.add_argument("--split",      type=str, default="validation")
    p.add_argument("--seeds",      type=str, default="0,1,2")
    p.add_argument("--output",     type=str, default="output/results/summary.json")
    return p.parse_args()


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",")]
    seeds    = [int(s.strip()) for s in args.seeds.split(",")]

    all_results = {}

    print("\n" + "=" * 80)
    print(f"{'Dataset':<22} {'ClaimAcc':>9} {'UnsuppF1':>9} {'ContrF1':>8} {'PathCorr':>9} {'ECE':>7}")
    print("-" * 80)

    for dataset in datasets:
        pred_dir = os.path.join(args.pred_root, dataset)
        if not os.path.isdir(pred_dir):
            print(f"{dataset:<22}  [MISSING: {pred_dir}]")
            continue

        per_seed_metrics = []
        for seed in seeds:
            try:
                m = evaluate_seed(pred_dir, dataset, args.split, seed)
                if m:
                    per_seed_metrics.append(m)
            except FileNotFoundError:
                pass

        if not per_seed_metrics:
            print(f"{dataset:<22}  [NO PREDICTIONS]")
            continue

        agg = {}
        for key in ["claim_acc", "unsupp_f1", "contr_f1", "path_corr", "ece"]:
            vals = [m[key] for m in per_seed_metrics if key in m]
            if vals:
                agg[key] = {"mean": np.mean(vals), "std": np.std(vals)}

        all_results[dataset] = agg

        def _v(k):
            return f"{agg[k]['mean']:.1f}±{agg[k]['std']:.2f}" if k in agg else "  -"

        print(
            f"{dataset:<22} "
            f"{_v('claim_acc'):>9} "
            f"{_v('unsupp_f1'):>9} "
            f"{_v('contr_f1'):>8} "
            f"{_v('path_corr'):>9} "
            f"{_v('ece'):>7}"
        )

    print("=" * 80)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
