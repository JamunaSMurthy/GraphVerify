"""
Runs GraphVerify on a dataset split and saves claim-level predictions.

The generated answer can come from:
  (a) a pre-generated answers file (--generated_answers)
  (b) the sample's "generated" field
  (c) the gold answer (for oracle upper-bound experiments)

Usage:
  python verify.py \\
      --dataset hotpotqa \\
      --split validation \\
      --graph_dir output/graphs/hotpotqa \\
      --output_dir output/predictions/hotpotqa \\
      --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
from tqdm import tqdm

from graphverify.config import GraphVerifyConfig
from graphverify.verifier import GraphVerify
from graphverify.evidence_graph import EvidenceGraph
from dataset.loader import load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",           type=str, required=True)
    p.add_argument("--split",             type=str, default="validation")
    p.add_argument("--data_dir",          type=str, default=None)
    p.add_argument("--graph_dir",         type=str, default=None)
    p.add_argument("--generated_answers", type=str, default=None,
                   help="JSONL with {id, generated_answer} per sample")
    p.add_argument("--output_dir",        type=str, default="output/predictions")
    p.add_argument("--max_samples",       type=int, default=None)
    p.add_argument("--seed",              type=int, default=0)
    p.add_argument("--llm_backend",       type=str, default="openai", choices=["openai", "local"])
    p.add_argument("--llm_model",         type=str, default="gpt-4o-mini")
    p.add_argument("--local_model",       type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--embed_model",       type=str, default="BAAI/bge-base-en-v1.5")
    p.add_argument("--evidence_mode",     type=str, default="hybrid",
                   choices=["text", "retrieved_graph", "kg_paths", "hybrid"])
    p.add_argument("--calibrator_path",   type=str, default=None)
    p.add_argument("--support_threshold",    type=float, default=None)
    p.add_argument("--contradict_threshold", type=float, default=None)
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_graphs(graph_dir: str, dataset: str, split: str) -> dict:
    path = os.path.join(graph_dir, f"{dataset}_{split}_graphs.jsonl")
    if not os.path.exists(path):
        print(f"[WARN] Graph file not found: {path}. Graphs will be built on the fly.")
        return {}
    graphs = {}
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                graphs[str(rec["id"])] = rec.get("graph", {})
            except Exception:
                pass
    print(f"  Loaded {len(graphs)} pre-built graphs.")
    return graphs


def load_generated_answers(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    ans = {}
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                ans[str(rec["id"])] = rec.get("generated_answer", rec.get("answer", ""))
            except Exception:
                pass
    return ans


def main():
    args = parse_args()
    seed_everything(args.seed)

    cfg_kwargs = dict(
        llm_backend=args.llm_backend, llm_model=args.llm_model,
        local_model_path=args.local_model, embed_model=args.embed_model,
        evidence_mode=args.evidence_mode,
    )
    if args.support_threshold is not None:
        cfg_kwargs["support_threshold"] = args.support_threshold
    if args.contradict_threshold is not None:
        cfg_kwargs["contradict_threshold"] = args.contradict_threshold

    gv = GraphVerify(GraphVerifyConfig(**cfg_kwargs))
    if args.calibrator_path:
        gv.load_calibrator(args.calibrator_path)

    print(f"Loading {args.dataset} [{args.split}]...")
    samples = load_dataset(args.dataset, split=args.split,
                           data_dir=args.data_dir, max_samples=args.max_samples)

    graphs_cache    = load_graphs(args.graph_dir, args.dataset, args.split) if args.graph_dir else {}
    generated_cache = load_generated_answers(args.generated_answers)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.dataset}_{args.split}_seed{args.seed}.jsonl")

    print(f"Verifying {len(samples)} samples → {out_path}\n")

    with open(out_path, "w") as fout:
        for sample in tqdm(samples, desc=f"[{args.dataset}]"):
            sid    = str(sample.get("id", ""))
            answer = (generated_cache.get(sid)
                      or sample.get("generated", "")
                      or sample.get("answer", ""))
            if not answer:
                continue

            graph = None
            if sid in graphs_cache:
                try:
                    graph = EvidenceGraph.from_dict(graphs_cache[sid])
                except Exception:
                    pass

            try:
                output = gv.verify(query=sample.get("query", ""),
                                   passages=sample.get("passages", []),
                                   answer=answer, graph=graph)
            except Exception as e:
                print(f"\n[WARN] {sid}: {e}")
                output = None

            record = {
                "id":           sid,
                "query":        sample.get("query", ""),
                "answer":       answer,
                "gold_verdict": sample.get("gold_verdict", ""),
                "gold_path":    sample.get("gold_path", ""),
                "seed":         args.seed,
                "records":      output.records     if output else [],
                "graph_stats":  output.graph_stats if output else {},
                "verdicts":     output.verdicts    if output else [],
            }
            fout.write(json.dumps(record) + "\n")

    print(f"\nSaved predictions to {out_path}")


if __name__ == "__main__":
    main()
