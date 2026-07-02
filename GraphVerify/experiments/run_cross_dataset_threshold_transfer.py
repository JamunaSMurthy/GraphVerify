"""
Cross-dataset threshold transfer: tunes the support/contradiction
thresholds on one dataset's dev split and evaluates the *same fixed*
thresholds on every other dataset's test split. This is the calibration-
overfitting defense the revision plan requires: if GraphVerify's thresholds
only work on the dataset they were tuned on, that is a sign of per-dataset
overfitting rather than a genuinely useful decision rule.

Reuses `experiments/run_label_efficiency_experiment.py`'s scoring/threshold
machinery (path scores are threshold-independent, so thresholds selected on
one dataset are just applied to another dataset's already-computed scores
with no re-running of the pipeline).

Usage:
  python experiments/run_cross_dataset_threshold_transfer.py \\
      --tune_on hotpotqa --eval_on 2wikimultihopqa,musique,fever,ragtruth \\
      --split validation --max_samples 200 \\
      --output output/results/threshold_transfer.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.answer_generation import generate_answers_for_dataset
from experiments._common import add_llm_args, build_config, build_llm_client, load_samples, parse_dataset_list, save_csv
from experiments.run_label_efficiency_experiment import compute_claim_scores, evaluate_thresholds, select_thresholds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tune_on", type=str, required=True, help="Dataset used to select thresholds.")
    p.add_argument("--eval_on", type=str, required=True, help="Comma-separated datasets to evaluate the fixed thresholds on.")
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    add_llm_args(p)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default="output/results/threshold_transfer.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    llm_client = build_llm_client(args)
    eval_datasets: List[str] = parse_dataset_list(args.eval_on)

    tune_samples = load_samples(args.tune_on, args)
    tune_samples = generate_answers_for_dataset(llm_client, tune_samples)
    tune_claims = compute_claim_scores(tune_samples, llm_client, cfg)
    support_threshold, contradict_threshold = select_thresholds(tune_claims, label_fraction=1.0, seed=args.seed)
    print(f"Tuned on {args.tune_on}: support_threshold={support_threshold}, contradict_threshold={contradict_threshold}")

    rows = [{
        "dataset": args.tune_on, "role": "tune", "support_threshold": support_threshold,
        "contradict_threshold": contradict_threshold,
        **evaluate_thresholds(tune_claims, support_threshold, contradict_threshold),
    }]

    for dataset in eval_datasets:
        eval_samples = load_samples(dataset, args)
        eval_samples = generate_answers_for_dataset(llm_client, eval_samples)
        eval_claims = compute_claim_scores(eval_samples, llm_client, cfg)
        metrics = evaluate_thresholds(eval_claims, support_threshold, contradict_threshold)
        rows.append({
            "dataset": dataset, "role": "transfer", "support_threshold": support_threshold,
            "contradict_threshold": contradict_threshold, **metrics,
        })

    print(f"\n{'Dataset':<18} {'Role':<10} {'ClaimAcc':>9} {'MacroF1':>8} {'ContrF1':>8}")
    for r in rows:
        print(f"{r['dataset']:<18} {r['role']:<10} {r.get('claim_acc', 0):>9.1f} {r.get('macro_f1', 0):>8.1f} {r.get('contr_f1', 0):>8.1f}")

    save_csv(rows, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
