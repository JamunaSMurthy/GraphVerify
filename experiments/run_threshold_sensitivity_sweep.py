"""
Threshold sensitivity sweep: grid over (support_threshold,
contradict_threshold), reporting Claim Accuracy, Unsupported F1,
Contradiction F1, and ECE for every combination. Defends against the
"thresholds were cherry-picked" criticism by showing the full response
surface, not just the single chosen operating point.

Reuses `experiments/run_label_efficiency_experiment.py:compute_claim_scores`
so the pipeline runs once and every grid point is evaluated by re-deriving
verdicts from cached scores.

Usage:
  python experiments/run_threshold_sensitivity_sweep.py \\
      --dataset hotpotqa --split validation --max_samples 200 \\
      --support_grid 0.4,0.5,0.6,0.7,0.8 \\
      --contradict_grid 0.4,0.5,0.6,0.7,0.8 \\
      --output output/results/threshold_sensitivity.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.answer_generation import generate_answers_for_dataset
from eval.metrics import compute_all_metrics
from experiments._common import add_llm_args, build_config, build_llm_client, load_samples, save_csv
from experiments.run_label_efficiency_experiment import compute_claim_scores
from graphverify.verdict_assigner import VerdictAssigner


def sweep_thresholds(
    claims: List[Dict[str, Any]],
    support_grid: List[float],
    contradict_grid: List[float],
) -> List[Dict[str, Any]]:
    rows = []
    golds = [c["gold_verdict"] for c in claims]
    for ts in support_grid:
        for tc in contradict_grid:
            assigner = VerdictAssigner(support_threshold=ts, contradict_threshold=tc)
            preds = [assigner.verdict_from_scores(c["support_score"], c["contradict_score"]) for c in claims]
            rel_scores = [max(c["support_score"], c["contradict_score"]) for c in claims]
            metrics = compute_all_metrics(preds, golds, rel_scores=rel_scores)
            rows.append({"support_threshold": ts, "contradict_threshold": tc, **metrics})
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    add_llm_args(p)
    p.add_argument("--support_grid", type=str, default="0.40,0.50,0.60,0.70,0.80")
    p.add_argument("--contradict_grid", type=str, default="0.40,0.50,0.60,0.70,0.80")
    p.add_argument("--output", type=str, default="output/results/threshold_sensitivity.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    llm_client = build_llm_client(args)

    samples = load_samples(args.dataset, args)
    samples = generate_answers_for_dataset(llm_client, samples)
    claims = compute_claim_scores(samples, llm_client, cfg)

    support_grid = [float(x) for x in args.support_grid.split(",")]
    contradict_grid = [float(x) for x in args.contradict_grid.split(",")]
    rows = sweep_thresholds(claims, support_grid, contradict_grid)

    print(f"{'theta_s':>8} {'theta_c':>8} {'ClaimAcc':>9} {'UnsuppF1':>9} {'ContrF1':>8} {'ECE':>7}")
    for r in rows:
        print(f"{r['support_threshold']:>8.2f} {r['contradict_threshold']:>8.2f} "
              f"{r.get('claim_acc', 0):>9.1f} {r.get('unsupp_f1', 0):>9.1f} {r.get('contr_f1', 0):>8.1f} {r.get('ece', 0):>7.3f}")

    save_csv(rows, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
