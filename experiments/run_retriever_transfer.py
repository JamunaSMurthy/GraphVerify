"""
Retriever transfer: re-ranks each sample's existing candidate passage pool
with a different embedding model (e.g. BGE-base vs. Contriever) and reruns
verification, to test whether results depend on which retriever surfaced
the evidence.

**Scope note.** This does not build a new corpus-wide retrieval index for
each retriever -- that requires per-dataset corpus infrastructure beyond
this benchmark's scope (see `docs/HARDWARE_SOFTWARE_REQUIREMENTS.md`).
Instead it re-embeds and re-ranks the *existing* candidate passages each
dataset loader already provides (the gold + distractor passages HotpotQA/
2Wiki/MuSiQue ship, or FEVER's evidence set) with the alternate embedding
model, and truncates to `--top_k`. This measures "does re-ranking the same
candidate pool with a different embedding model change results" -- a
meaningful, honest retriever-sensitivity check -- rather than "what would a
completely different retrieval system have surfaced from the full corpus."

Reports each retriever's claim-level metrics plus "retrieved-support
recall": the fraction of samples where a passage whose title appears in the
sample's `gold_path` annotation survived into the re-ranked top-k (0 for
datasets without a title-bearing `gold_path`).

Usage:
  python experiments/run_retriever_transfer.py \\
      --dataset hotpotqa --split validation --max_samples 100 \\
      --retrievers BAAI/bge-base-en-v1.5,facebook/contriever \\
      --top_k 5 --methods graphverify_hybrid \\
      --output output/results/retriever_transfer.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.answer_generation import generate_answers_for_dataset
from experiments._common import build_llm_client, load_samples, parse_dataset_list, save_csv
from experiments.run_main_verification_benchmark import run_benchmark_on_dataset
from graphverify.config import GraphVerifyConfig


def rerank_passages(passages: List[Dict[str, Any]], query: str, embed_model: str, top_k: int) -> List[Dict[str, Any]]:
    from graphverify.embedder import Embedder
    if not passages:
        return []
    embedder = Embedder(embed_model)
    query_vec = embedder.encode([query])
    passage_vecs = embedder.encode([p.get("text", "") for p in passages])
    sims = embedder.cosine_sim_matrix(query_vec, passage_vecs)[0]
    ranked_idx = sims.argsort()[::-1][:top_k]

    reranked = []
    for rank, idx in enumerate(ranked_idx, start=1):
        p = dict(passages[int(idx)])
        p["rank"] = rank
        p["score"] = float(sims[int(idx)])
        reranked.append(p)
    return reranked


def retrieved_support_recall(samples: List[Dict[str, Any]]) -> float:
    """Fraction of samples whose top-ranked passage titles cover a title mentioned in `gold_path`."""
    scored = [s for s in samples if s.get("gold_path")]
    if not scored:
        return float("nan")
    hits = 0
    for s in scored:
        titles_in_pool = {p.get("title", "") for p in s.get("passages", [])}
        if any(title and title in s["gold_path"] for title in titles_in_pool):
            hits += 1
    return 100.0 * hits / len(scored)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--retrievers", type=str, default="BAAI/bge-base-en-v1.5,facebook/contriever")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--methods", type=str, default="graphverify_hybrid,llm_text_verifier")
    p.add_argument("--llm_backend", type=str, default="openai", choices=["openai", "anthropic", "local"])
    p.add_argument("--llm_model", type=str, default="gpt-4o-mini")
    p.add_argument("--local_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--embed_model", type=str, default="BAAI/bge-base-en-v1.5")
    p.add_argument("--output", type=str, default="output/results/retriever_transfer.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = GraphVerifyConfig(llm_backend=args.llm_backend, llm_model=args.llm_model, local_model_path=args.local_model, embed_model=args.embed_model)
    llm_client = build_llm_client(args)
    method_names = parse_dataset_list(args.methods)
    retrievers = parse_dataset_list(args.retrievers)

    base_samples = load_samples(args.dataset, args)
    base_samples = generate_answers_for_dataset(llm_client, base_samples)

    rows = []
    for retriever_model in retrievers:
        reranked_samples = []
        for sample in base_samples:
            s = dict(sample)
            s["passages"] = rerank_passages(sample.get("passages", []), sample.get("query", ""), retriever_model, args.top_k)
            reranked_samples.append(s)

        recall = retrieved_support_recall(reranked_samples)
        dataset_results = run_benchmark_on_dataset(args.dataset, reranked_samples, method_names, llm_client, cfg)

        print(f"\n=== retriever: {retriever_model} (top_k={args.top_k}, retrieved-support recall={recall:.1f}%) ===")
        for method_name, result in dataset_results.items():
            m = result["metrics"]
            rows.append({"retriever": retriever_model, "dataset": args.dataset, "method": method_name,
                         "retrieved_support_recall": recall, **m})
            print(f"  {method_name:<20} claim_acc={m.get('claim_acc', 0):.1f} contr_f1={m.get('contr_f1', 0):.1f}")

    save_csv(rows, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
