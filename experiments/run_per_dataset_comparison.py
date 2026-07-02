"""
Per-dataset comparison: for each dataset, reports every method's claim-level
metrics side by side, identifies the strongest baseline (by claim
accuracy), and runs a paired bootstrap significance test between
GraphVerify-hybrid and that strongest baseline.

This is the check the revision plan calls "non-negotiable for credibility":
an aggregate-only main table can hide a dataset where GraphVerify actually
underperforms. Reusing this script's output alongside
`run_main_verification_benchmark.py`'s aggregate table lets a reviewer see
whether a claimed gain holds per-dataset or is an average-hiding artifact.

Usage:
  python experiments/run_per_dataset_comparison.py \\
      --datasets hotpotqa,2wikimultihopqa,musique,fever,ragtruth \\
      --split validation --max_samples 200 \\
      --output_dir output/results/per_dataset_comparison
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.answer_generation import generate_answers_for_dataset
from eval.significance import paired_bootstrap_significance
from experiments._common import add_dataset_args, add_llm_args, build_config, build_llm_client, load_samples, parse_dataset_list, save_csv, save_json
from experiments._methods import ALL_METHOD_NAMES, GRAPHVERIFY_METHOD_NAMES
from experiments.run_main_verification_benchmark import run_benchmark_on_dataset


def strongest_baseline(dataset_results: Dict[str, Dict[str, Any]], metric: str = "claim_acc") -> str:
    """Returns the baseline method name (excluding GraphVerify variants) with the highest `metric`."""
    baseline_names = [n for n in dataset_results if n not in GRAPHVERIFY_METHOD_NAMES]
    if not baseline_names:
        raise ValueError("No baseline methods present in dataset_results.")
    return max(baseline_names, key=lambda n: dataset_results[n]["metrics"].get(metric, 0.0))


def _claim_arrays(predictions: List[Dict[str, Any]]):
    preds, golds, cluster_ids, item_ids = [], [], [], []
    for sample in predictions:
        for i, rec in enumerate(sample.get("records", [])):
            preds.append(rec.get("verdict", "Unsupported"))
            golds.append(sample.get("gold_verdict", "Unsupported"))
            cluster_ids.append(sample["id"])
            item_ids.append(f"{sample['id']}::{i}")
    return preds, golds, cluster_ids, item_ids


def compare_dataset(dataset_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    best_baseline = strongest_baseline(dataset_results)
    gv_preds, gv_golds, gv_clusters, gv_ids = _claim_arrays(dataset_results["graphverify_hybrid"]["predictions"])
    bl_preds, bl_golds, bl_clusters, bl_ids = _claim_arrays(dataset_results[best_baseline]["predictions"])

    # Align by shared claim id (both systems verified the same shared claims per sample).
    bl_by_id = dict(zip(bl_ids, bl_preds))
    common_ids = [i for i in gv_ids if i in bl_by_id]
    aligned_a = [dict(zip(gv_ids, gv_preds))[i] for i in common_ids]
    aligned_b = [bl_by_id[i] for i in common_ids]
    aligned_golds = [dict(zip(gv_ids, gv_golds))[i] for i in common_ids]
    aligned_clusters = [i.split("::")[0] for i in common_ids]

    significance = None
    if common_ids:
        result = paired_bootstrap_significance(
            aligned_a, aligned_b, aligned_golds, metric="claim_acc", cluster_ids=aligned_clusters,
        )
        significance = {
            "strongest_baseline": best_baseline,
            "graphverify_hybrid_score": result.system_a_score,
            "baseline_score": result.system_b_score,
            "effect_size": result.effect_size,
            "p_value": result.p_value,
            "ci_low": result.ci_low,
            "ci_high": result.ci_high,
        }

    return {"strongest_baseline": best_baseline, "significance": significance}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(p)
    add_llm_args(p)
    p.add_argument("--methods", type=str, default=",".join(ALL_METHOD_NAMES))
    p.add_argument("--output_dir", type=str, default="output/results/per_dataset_comparison")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    datasets = parse_dataset_list(args.datasets)
    method_names = parse_dataset_list(args.methods)
    cfg = build_config(args)
    llm_client = build_llm_client(args)

    all_rows = []
    for dataset in datasets:
        samples = load_samples(dataset, args)
        samples = generate_answers_for_dataset(llm_client, samples)
        dataset_results = run_benchmark_on_dataset(dataset, samples, method_names, llm_client, cfg)

        comparison = compare_dataset(dataset_results) if "graphverify_hybrid" in dataset_results else {}
        print(f"\n=== {dataset} ===")
        print(f"{'Method':<20} {'ClaimAcc':>9} {'MacroF1':>8} {'ContrF1':>8} {'PathCorr':>9}")
        for name, result in dataset_results.items():
            m = result["metrics"]
            row = {"dataset": dataset, "method": name, **m}
            all_rows.append(row)
            print(f"{name:<20} {m.get('claim_acc', 0):>9.1f} {m.get('macro_f1', 0):>8.1f} "
                  f"{m.get('contr_f1', 0):>8.1f} {m.get('path_corr', 0):>9.1f}")
        if comparison.get("significance"):
            sig = comparison["significance"]
            print(f"  strongest baseline: {sig['strongest_baseline']} "
                  f"(GraphVerify-hybrid {sig['graphverify_hybrid_score']:.1f} vs. baseline {sig['baseline_score']:.1f}, "
                  f"effect={sig['effect_size']:+.1f}, p={sig['p_value']:.4f})")
            save_json(comparison, os.path.join(args.output_dir, dataset, "significance.json"))

    save_csv(all_rows, os.path.join(args.output_dir, "per_dataset_metrics.csv"))
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
