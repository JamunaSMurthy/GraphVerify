"""
Computes claim-level metrics from GraphVerify predictions.

Reports mean ± std across multiple verifier seeds.

Usage:
  python evaluate.py \\
      --pred_dir output/predictions/hotpotqa \\
      --dataset hotpotqa \\
      --split validation \\
      --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from eval.metrics import compute_all_metrics, run_bootstrap


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir",  type=str, required=True)
    p.add_argument("--dataset",   type=str, required=True)
    p.add_argument("--split",     type=str, default="validation")
    p.add_argument("--seeds",     type=str, default="0",
                   help="Comma-separated seeds, e.g. 0,1,2")
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--output",    type=str, default=None)
    return p.parse_args()


def load_predictions(pred_dir: str, dataset: str, split: str, seed: int) -> List[Dict]:
    path = os.path.join(pred_dir, f"{dataset}_{split}_seed{seed}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prediction file not found: {path}")
    preds = []
    with open(path) as f:
        for line in f:
            try:
                preds.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return preds


def aggregate_claim_labels(predictions: List[Dict]) -> Tuple[List, List, List, List, List]:
    """Flatten per-claim records into parallel lists for metric computation."""
    pred_verdicts, gold_verdicts = [], []
    pred_paths,    gold_paths    = [], []
    rel_scores                   = []

    for sample in predictions:
        gold_verdict = sample.get("gold_verdict", "")
        gold_path    = sample.get("gold_path",    "")
        for rec in sample.get("records", []):
            pred_verdicts.append(rec.get("verdict", "Unsupported"))
            gold_verdicts.append(gold_verdict or "Unsupported")
            pred_paths.append(_path_to_str(rec.get("best_path")))
            gold_paths.append(gold_path)
            rel_scores.append(float(rec.get("reliability", 0.0)))

    return pred_verdicts, gold_verdicts, pred_paths, gold_paths, rel_scores


def _path_to_str(path) -> str:
    if not path:
        return ""
    if isinstance(path, str):
        return path
    if isinstance(path, list):
        parts = []
        for e in path:
            if isinstance(e, dict):
                parts.append(
                    f"{e.get('src_label', e.get('src',''))} → "
                    f"{e.get('relation','')} → "
                    f"{e.get('dst_label', e.get('dst',''))}"
                )
        return "; ".join(parts)
    return ""


def evaluate_seed(pred_dir: str, dataset: str, split: str, seed: int) -> Dict:
    predictions = load_predictions(pred_dir, dataset, split, seed)
    pv, gv, pp, gp, rs = aggregate_claim_labels(predictions)
    if not pv:
        print(f"[WARN] No claim records found for seed {seed}.")
        return {}
    metrics = compute_all_metrics(pv, gv, pred_paths=pp, gold_paths=gp, rel_scores=rs)
    metrics["n_claims"] = len(pv)
    return metrics


def main():
    args  = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    per_seed: List[Dict] = []
    for seed in seeds:
        try:
            m = evaluate_seed(args.pred_dir, args.dataset, args.split, seed)
            per_seed.append(m)
            parts = "  ".join(f"{k}={v:.2f}" for k, v in sorted(m.items())
                              if k != "n_claims" and isinstance(v, float))
            print(f"Seed {seed}: {parts}")
        except FileNotFoundError as e:
            print(f"Seed {seed}: {e}")

    if not per_seed:
        print("No results.")
        return

    all_keys = {k for m in per_seed for k in m} - {"n_claims"}

    print("\n" + "=" * 55)
    print(f"{'Metric':<20} {'Mean':>8}  {'±Std':>8}")
    print("-" * 55)
    aggregated: Dict = {}
    for k in sorted(all_keys):
        vals = [m[k] for m in per_seed if k in m]
        if not vals:
            continue
        mean, std = float(np.mean(vals)), float(np.std(vals))
        aggregated[k] = {"mean": mean, "std": std, "values": vals}
        print(f"{k:<20} {mean:>8.2f}  ±{std:>6.3f}")
    print("=" * 55)

    if args.bootstrap:
        try:
            preds = load_predictions(args.pred_dir, args.dataset, args.split, seeds[0])
            pv, gv, _, _, _ = aggregate_claim_labels(preds)
            if pv:
                print("\nPaired bootstrap (seed 0):")
                for metric in ["claim_acc", "unsupp_f1", "contr_f1"]:
                    pt, lo, hi = run_bootstrap(pv, gv, metric=metric)
                    print(f"  {metric}: {pt:.2f} [{lo:.2f}, {hi:.2f}]")
        except Exception as e:
            print(f"Bootstrap failed: {e}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"dataset": args.dataset, "split": args.split,
                       "seeds": seeds, "metrics": aggregated, "per_seed": per_seed}, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
