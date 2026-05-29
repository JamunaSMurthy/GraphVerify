"""
Component ablation runner.

Evaluates each ablation variant by overriding GraphVerifyConfig fields
and running the full verification pipeline on a sample subset.

Usage:
  python eval/ablation.py \\
      --dataset hotpotqa \\
      --split validation \\
      --data_dir path/to/data \\
      --graph_dir output/graphs/hotpotqa \\
      --llm_backend openai \\
      --output output/ablation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from graphverify.config import GraphVerifyConfig
from graphverify.verifier import GraphVerify
from graphverify.evidence_graph import EvidenceGraph
from dataset.loader import load_dataset
from eval.metrics import compute_all_metrics


@dataclass
class AblationVariant:
    name:   str
    kwargs: Dict  # GraphVerifyConfig overrides


ABLATION_VARIANTS: List[AblationVariant] = [
    AblationVariant("w/o claim decomp.",    {"max_claims_override": 1}),
    AblationVariant("w/o relation norm.",   {"skip_relation_norm": True}),
    AblationVariant("w/o provenance",       {"lambda_prov": 0.0}),
    AblationVariant("w/o contradiction",    {"contradict_threshold": 1.01}),
    AblationVariant("exact matching only",  {"embed_cosine_cutoff": 1.0}),
    AblationVariant("semantic only",        {"exact_match_only": False, "alias_match_only": False}),
    AblationVariant("w/o hybrid evidence",  {"evidence_mode": "retrieved_graph"}),
    AblationVariant("Full GraphVerify",     {}),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",     type=str, required=True)
    p.add_argument("--split",       type=str, default="validation")
    p.add_argument("--data_dir",    type=str, default=None)
    p.add_argument("--graph_dir",   type=str, default=None)
    p.add_argument("--llm_backend", type=str, default="openai")
    p.add_argument("--llm_model",   type=str, default="gpt-4o-mini")
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--output",      type=str, default="output/ablation")
    p.add_argument("--seed",        type=int, default=0)
    return p.parse_args()


def run_variant(
    variant: AblationVariant,
    samples: List[Dict],
    graphs_cache: Dict,
    base_cfg: GraphVerifyConfig,
) -> Dict:
    valid_fields = set(GraphVerifyConfig.__dataclass_fields__.keys())
    cfg_dict = {
        k: getattr(base_cfg, k)
        for k in ["llm_backend", "llm_model", "embed_model",
                  "contradict_threshold", "support_threshold",
                  "embed_cosine_cutoff", "lambda_prov", "evidence_mode"]
    }
    for k, v in variant.kwargs.items():
        if k in valid_fields:
            cfg_dict[k] = v

    cfg = GraphVerifyConfig(**{k: v for k, v in cfg_dict.items() if k in valid_fields})
    gv  = GraphVerify(cfg)

    preds, golds, pred_paths, gold_paths, scores = [], [], [], [], []

    for sample in samples:
        sid = str(sample.get("id", ""))
        ans = sample.get("answer", sample.get("generated", ""))
        if not ans:
            continue

        graph = None
        if sid in graphs_cache:
            try:
                graph = EvidenceGraph.from_dict(graphs_cache[sid])
            except Exception:
                pass

        try:
            out  = gv.verify(query=sample.get("query", ""),
                             passages=sample.get("passages", []),
                             answer=ans, graph=graph)
            gold = sample.get("gold_verdict", "Unsupported")
            for rec in out.records:
                preds.append(rec.get("verdict", "Unsupported"))
                golds.append(gold)
                pred_paths.append("")
                gold_paths.append(sample.get("gold_path", ""))
                scores.append(float(rec.get("reliability", 0.0)))
        except Exception as e:
            print(f"  [WARN] {sid}: {e}")

    if not preds:
        return {}
    return compute_all_metrics(preds, golds, pred_paths=pred_paths,
                               gold_paths=gold_paths, rel_scores=scores)


def main():
    args = parse_args()

    base_cfg = GraphVerifyConfig(llm_backend=args.llm_backend, llm_model=args.llm_model)
    samples  = load_dataset(args.dataset, split=args.split, data_dir=args.data_dir,
                            max_samples=args.max_samples)

    graphs_cache: Dict = {}
    if args.graph_dir:
        gpath = os.path.join(args.graph_dir, f"{args.dataset}_{args.split}_graphs.jsonl")
        if os.path.exists(gpath):
            with open(gpath) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        graphs_cache[str(r["id"])] = r.get("graph", {})
                    except Exception:
                        pass

    print(f"\nAblation on {args.dataset} [{args.split}], {len(samples)} samples")
    print("=" * 72)
    print(f"{'Variant':<32} {'Claim':>7} {'Unsupp':>7} {'Contr':>7} {'Path':>7}")
    print("-" * 72)

    results = {}
    for variant in ABLATION_VARIANTS:
        print(f"{variant.name:<32}", end=" ", flush=True)
        m = run_variant(variant, samples, graphs_cache, base_cfg)
        results[variant.name] = m
        if m:
            print(f"{m.get('claim_acc',0):>7.1f} {m.get('unsupp_f1',0):>7.1f} "
                  f"{m.get('contr_f1',0):>7.1f} {m.get('path_corr',0):>7.1f}")
        else:
            print("  [no results]")

    print("=" * 72)

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, f"{args.dataset}_ablation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
