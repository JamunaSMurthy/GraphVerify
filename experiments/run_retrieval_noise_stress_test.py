"""
Retrieval-noise stress test: re-runs verification under five controlled
passage perturbations (`dataset/stress_test.py`) and compares each method's
metric degradation, to test whether provenance-linked graph evidence is
actually more robust to noisy retrieval than a text-only verifier, rather
than just scoring higher on clean retrieval.

Conditions: clean (baseline), top-k halved, distractor passages injected,
bridge evidence removed (multi-hop datasets only, requires `bridge_titles`
per sample -- see `dataset/stress_test.remove_bridge_evidence`), entity
alias noise, numeric/date corruption.

Usage:
  python experiments/run_retrieval_noise_stress_test.py \\
      --datasets hotpotqa,2wikimultihopqa \\
      --split validation --max_samples 200 \\
      --methods graphverify_hybrid,llm_text_verifier \\
      --output_dir output/results/stress_test
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.answer_generation import generate_answers_for_dataset
from dataset.stress_test import (
    corrupt_numeric_and_date_mentions,
    inject_distractor_passages,
    inject_entity_alias_noise,
    perturb_top_k,
    remove_bridge_evidence,
)
from experiments._common import add_dataset_args, add_llm_args, build_config, build_llm_client, load_samples, parse_dataset_list, save_csv
from experiments._methods import build_method
from graphverify.claim_decomposer import ClaimDecomposer
from eval.metrics import compute_all_metrics

CONDITIONS = ("clean", "top_k_halved", "distractors", "bridge_removed", "entity_alias_noise", "numeric_date_corruption")


def build_distractor_pool(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flattens every sample's passages into one pool usable as distractors for a *different* sample."""
    return [p for s in samples for p in s.get("passages", [])]


def apply_condition(
    condition: str,
    sample: Dict[str, Any],
    distractor_pool: List[Dict[str, Any]],
    seed: int,
) -> List[Dict[str, Any]]:
    passages = sample.get("passages", [])
    if condition == "clean":
        return passages
    if condition == "top_k_halved":
        return perturb_top_k(passages, max(1, len(passages) // 2))
    if condition == "distractors":
        return inject_distractor_passages(passages, distractor_pool, n=2, seed=seed)
    if condition == "bridge_removed":
        bridge_titles = sample.get("bridge_titles") or []
        if not bridge_titles:
            return passages  # no annotated bridge title for this sample -- condition not applicable
        return remove_bridge_evidence(passages, bridge_titles)
    if condition == "entity_alias_noise":
        return inject_entity_alias_noise(passages, seed=seed)
    if condition == "numeric_date_corruption":
        return corrupt_numeric_and_date_mentions(passages, seed=seed)
    raise ValueError(f"Unknown stress condition: {condition}")


def run_stress_test_on_dataset(
    dataset: str,
    samples: List[Dict[str, Any]],
    method_names: List[str],
    llm_client,
    cfg,
    conditions: List[str] = CONDITIONS,
    seed: int = 0,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Returns {condition: {method_name: metrics_dict}}."""
    decomposer = ClaimDecomposer(llm_client)
    claims_by_id = {str(s["id"]): decomposer.decompose(s.get("generated", "")) for s in samples}
    distractor_pool = build_distractor_pool(samples)

    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for condition in conditions:
        results[condition] = {}
        for method_name in method_names:
            method = build_method(method_name, llm_client, cfg)
            preds, golds, cluster_ids = [], [], []
            for i, sample in enumerate(samples):
                sid = str(sample["id"])
                claims = claims_by_id[sid]
                if not claims:
                    continue
                perturbed_passages = apply_condition(condition, sample, distractor_pool, seed=seed + i)
                records = method.verify(sample.get("query", ""), perturbed_passages, claims)
                for rec in records:
                    preds.append(rec.get("verdict", "Unsupported"))
                    golds.append(sample.get("gold_verdict", "Unsupported"))
                    cluster_ids.append(sid)
            results[condition][method_name] = compute_all_metrics(preds, golds) if preds else {}
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(p)
    add_llm_args(p)
    p.add_argument("--methods", type=str, default="graphverify_hybrid,graphverify_score,llm_text_verifier")
    p.add_argument("--conditions", type=str, default=",".join(CONDITIONS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="output/results/stress_test")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    datasets = parse_dataset_list(args.datasets)
    method_names = parse_dataset_list(args.methods)
    conditions = parse_dataset_list(args.conditions)
    cfg = build_config(args)
    llm_client = build_llm_client(args)

    all_rows = []
    for dataset in datasets:
        samples = load_samples(dataset, args)
        samples = generate_answers_for_dataset(llm_client, samples)
        results = run_stress_test_on_dataset(dataset, samples, method_names, llm_client, cfg, conditions, seed=args.seed)

        print(f"\n=== {dataset} ===")
        print(f"{'Condition':<22} {'Method':<20} {'ClaimAcc':>9} {'ContrF1':>8}")
        for condition, per_method in results.items():
            for method_name, m in per_method.items():
                all_rows.append({"dataset": dataset, "condition": condition, "method": method_name, **m})
                print(f"{condition:<22} {method_name:<20} {m.get('claim_acc', 0):>9.1f} {m.get('contr_f1', 0):>8.1f}")

    save_csv(all_rows, os.path.join(args.output_dir, "stress_test_results.csv"))
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
