"""
Generator transfer: regenerates answers with two or more different LLM
backbones and reruns claim-level verification for each, holding the
verifier's own LLM fixed, to test whether a method's gains hold across
generator backbones rather than being specific to one generator's error
patterns.

Usage:
  python experiments/run_generator_transfer.py \\
      --dataset hotpotqa --split validation --max_samples 100 \\
      --generators openai:gpt-4o-mini,anthropic:claude-3-5-haiku-20241022 \\
      --methods graphverify_hybrid,llm_text_verifier \\
      --verifier_llm_backend openai --verifier_llm_model gpt-4o-mini \\
      --output output/results/generator_transfer.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.answer_generation import generate_answers_for_dataset
from experiments._common import load_samples, parse_dataset_list, save_csv
from experiments._methods import ALL_METHOD_NAMES
from experiments.run_main_verification_benchmark import run_benchmark_on_dataset
from graphverify.config import GraphVerifyConfig
from graphverify.llm_client import LLMClient


def run_generator_transfer(
    dataset: str,
    samples_template: List[Dict[str, Any]],
    generator_specs: List[str],
    method_names: List[str],
    verifier_llm_client,
    verifier_cfg,
) -> List[Dict[str, Any]]:
    """
    For each `"backend:model"` spec in `generator_specs`, regenerates every
    sample's answer with that backend/model and reruns every method in
    `method_names` (using the fixed `verifier_llm_client`/`verifier_cfg` for
    verification -- only the generator varies).
    """
    rows = []
    for spec in generator_specs:
        backend, model = spec.split(":", 1)
        gen_llm_client = LLMClient(GraphVerifyConfig(llm_backend=backend, llm_model=model))

        samples = [dict(s, generated="") for s in samples_template]
        samples = generate_answers_for_dataset(gen_llm_client, samples)

        dataset_results = run_benchmark_on_dataset(dataset, samples, method_names, verifier_llm_client, verifier_cfg)
        for method_name, result in dataset_results.items():
            rows.append({"generator": spec, "dataset": dataset, "method": method_name, **result["metrics"]})
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--generators", type=str, required=True, help="Comma-separated backend:model specs.")
    p.add_argument("--methods", type=str, default=",".join(ALL_METHOD_NAMES))
    p.add_argument("--verifier_llm_backend", type=str, default="openai")
    p.add_argument("--verifier_llm_model", type=str, default="gpt-4o-mini")
    p.add_argument("--output", type=str, default="output/results/generator_transfer.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    verifier_cfg = GraphVerifyConfig(llm_backend=args.verifier_llm_backend, llm_model=args.verifier_llm_model)
    verifier_llm_client = LLMClient(verifier_cfg)

    samples_template = load_samples(args.dataset, args)
    generator_specs = parse_dataset_list(args.generators)
    method_names = parse_dataset_list(args.methods)

    rows = run_generator_transfer(args.dataset, samples_template, generator_specs, method_names, verifier_llm_client, verifier_cfg)

    print(f"{'Generator':<40} {'Method':<20} {'ClaimAcc':>9} {'ContrF1':>8}")
    for r in rows:
        print(f"{r['generator']:<40} {r['method']:<20} {r.get('claim_acc', 0):>9.1f} {r.get('contr_f1', 0):>8.1f}")

    save_csv(rows, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
