"""
Builds and caches provenance-linked evidence graphs for a dataset split.

Graphs are saved as JSONL (one JSON object per line) and can be loaded
during verification to skip the expensive graph-building step.

Usage:
  python graph_build.py \\
      --dataset hotpotqa \\
      --split validation \\
      --output_dir output/graphs/hotpotqa \\
      --llm_backend openai \\
      --llm_model gpt-4o-mini \\
      --max_samples 500
"""
from __future__ import annotations

import argparse
import json
import os
from tqdm import tqdm

from graphverify.config import GraphVerifyConfig
from graphverify.llm_client import LLMClient
from graphverify.relation_normalizer import RelationNormalizer
from graphverify.evidence_graph import EvidenceGraphBuilder
from dataset.loader import load_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Build GraphVerify evidence graphs")
    p.add_argument("--dataset",     type=str, required=True)
    p.add_argument("--split",       type=str, default="validation")
    p.add_argument("--output_dir",  type=str, default="output/graphs")
    p.add_argument("--data_dir",    type=str, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--llm_backend", type=str, default="openai", choices=["openai", "local"])
    p.add_argument("--llm_model",   type=str, default="gpt-4o-mini")
    p.add_argument("--local_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--embed_model", type=str, default="BAAI/bge-base-en-v1.5")
    p.add_argument("--start_idx",   type=int, default=0,
                   help="Resume from this index (skip earlier samples)")
    return p.parse_args()


def main():
    args = parse_args()

    cfg      = GraphVerifyConfig(llm_backend=args.llm_backend, llm_model=args.llm_model,
                                  local_model_path=args.local_model, embed_model=args.embed_model)
    llm      = LLMClient(cfg)
    rel_norm = RelationNormalizer(embed_model=cfg.embed_model)
    builder  = EvidenceGraphBuilder(llm, rel_norm, embed_model=cfg.embed_model)

    print(f"Loading {args.dataset} [{args.split}]...")
    samples = load_dataset(args.dataset, split=args.split,
                           data_dir=args.data_dir, max_samples=args.max_samples)
    print(f"  Loaded {len(samples)} samples.")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.dataset}_{args.split}_graphs.jsonl")

    done_ids: set = set()
    if os.path.exists(out_path) and args.start_idx == 0:
        with open(out_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line).get("id", ""))
                except json.JSONDecodeError:
                    pass
        if done_ids:
            print(f"  Resuming — {len(done_ids)} graphs already built.")

    written = 0
    with open(out_path, "a" if done_ids else "w") as fout:
        for i, sample in enumerate(tqdm(samples, desc="Building graphs")):
            if i < args.start_idx:
                continue
            sid = str(sample.get("id", i))
            if sid in done_ids:
                continue

            passages = sample.get("passages", [])
            if not passages:
                fout.write(json.dumps({"id": sid, "graph": {"nodes": [], "edges": []}, "n_passages": 0}) + "\n")
                continue

            try:
                graph  = builder.build(query=sample.get("query", ""), passages=passages)
                record = {"id": sid, "graph": graph.to_dict(),
                          "n_nodes": len(graph),
                          "n_edges": graph.nx_graph().number_of_edges(),
                          "n_passages": len(passages)}
            except Exception as e:
                print(f"\n[WARN] Sample {sid}: {e}")
                record = {"id": sid, "graph": {"nodes": [], "edges": []}, "error": str(e)}

            fout.write(json.dumps(record) + "\n")
            written += 1

    print(f"\nDone. Wrote {written} graph records to {out_path}")


if __name__ == "__main__":
    main()
