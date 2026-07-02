"""
Oracle pipeline decomposition: measures GraphVerify's upper bound by
bypassing individual pipeline stages with user-supplied gold artifacts,
isolating verifier-quality loss (path scoring / threshold assignment) from
pipeline-stage loss (claim decomposition, entity linking, relation
normalization, graph extraction).

Each dataset sample may optionally carry, beyond the unified schema
(`dataset/loader.py`):
  gold_claims : List[str]                             -- bypasses ClaimDecomposer
  gold_graph  : {"nodes": [...], "edges": [...]}       -- bypasses EvidenceGraphBuilder
                (graphverify.evidence_graph.EvidenceGraph.to_dict() schema)

Both are wired through :meth:`graphverify.verifier.GraphVerify.verify`'s
native `claims=`/`graph=` parameters, so no verifier code changes were
needed to support this. **None of the datasets loaded by
`dataset/loader.py` carry these fields today** -- they only exist once you
have run the human annotation pipeline (`dataset/claim_annotation.py`) and
merged its output back into a sample dict under these keys. This script
reports how many samples in the input actually had gold artifacts available
(`n_gold_claims_available`/`n_gold_graph_available`) so a 0 in that column
makes clear a variant ran with no real oracle advantage, rather than
silently falling back and reporting misleadingly-labeled "oracle" numbers.

Usage:
  python experiments/run_oracle_pipeline_decomposition.py \\
      --dataset hotpotqa --split validation --max_samples 200 \\
      --data_dir path/to/data_with_gold_claims_and_graph \\
      --output output/results/oracle_pipeline.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.answer_generation import generate_answers_for_dataset
from eval.metrics import compute_all_metrics
from experiments._common import add_llm_args, build_config, build_llm_client, load_samples, save_csv
from graphverify.evidence_graph import EvidenceGraph
from graphverify.verifier import build_graphverify

VARIANTS: List[Tuple[str, bool, bool]] = [
    ("full_pipeline",    False, False),
    ("gold_claims_only", True,  False),
    ("gold_graph_only",  False, True),
    ("full_oracle",      True,  True),
]


def load_gold_graph(sample: Dict[str, Any]) -> Optional[EvidenceGraph]:
    gold = sample.get("gold_graph")
    if not gold:
        return None
    return EvidenceGraph.from_dict(gold)


def run_oracle_variant(
    samples: List[Dict[str, Any]],
    llm_client,
    cfg,
    use_gold_claims: bool,
    use_gold_graph: bool,
) -> Dict[str, Any]:
    verifier = build_graphverify(cfg, llm_client=llm_client)
    preds, golds = [], []

    for sample in samples:
        answer = sample.get("generated") or sample.get("answer", "")
        if not answer:
            continue
        claims = sample.get("gold_claims") if (use_gold_claims and sample.get("gold_claims")) else None
        graph = load_gold_graph(sample) if use_gold_graph else None
        out = verifier.verify(query=sample.get("query", ""), passages=sample.get("passages", []), answer=answer, graph=graph, claims=claims)
        for rec in out.records:
            preds.append(rec["verdict"])
            golds.append(sample.get("gold_verdict", "Unsupported"))

    metrics = compute_all_metrics(preds, golds) if preds else {}
    metrics["n_gold_claims_available"] = sum(1 for s in samples if s.get("gold_claims"))
    metrics["n_gold_graph_available"] = sum(1 for s in samples if s.get("gold_graph"))
    metrics["n_samples"] = len(samples)
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    add_llm_args(p)
    p.add_argument("--output", type=str, default="output/results/oracle_pipeline.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    llm_client = build_llm_client(args)

    samples = load_samples(args.dataset, args)
    samples = generate_answers_for_dataset(llm_client, samples)

    rows = []
    print(f"{'Variant':<20} {'ClaimAcc':>9} {'MacroF1':>8} {'ContrF1':>8} {'GoldClaims':>11} {'GoldGraph':>10}")
    for name, use_claims, use_graph in VARIANTS:
        metrics = run_oracle_variant(samples, llm_client, cfg, use_claims, use_graph)
        rows.append({"variant": name, "dataset": args.dataset, **metrics})
        print(f"{name:<20} {metrics.get('claim_acc', 0):>9.1f} {metrics.get('macro_f1', 0):>8.1f} "
              f"{metrics.get('contr_f1', 0):>8.1f} {metrics.get('n_gold_claims_available', 0):>11} "
              f"{metrics.get('n_gold_graph_available', 0):>10}")

    save_csv(rows, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
