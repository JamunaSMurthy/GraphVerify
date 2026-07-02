"""
Runtime and compute benchmark: measures wall-clock latency (seconds per
answer, seconds per claim) and process memory for each verification
method, plus its cost relative to the base RAG pipeline alone (answer
generation with no verification step) -- demonstrating the method is
practical to run, not just accurate.

Memory is process resident-set-size (RSS) via `psutil`, sampled before and
after each benchmark; this measures *this process's* memory growth, not a
clean per-call allocation profile, since Python/embedding-model/HTTP-client
memory is not trivially isolated per call. It is a practical planning
number (matching what `docs/HARDWARE_SOFTWARE_REQUIREMENTS.md` needs), not
a memory-profiler-grade measurement.

Usage:
  python experiments/benchmark_runtime_and_compute.py \\
      --dataset hotpotqa --split validation --max_samples 50 \\
      --methods graphverify_score,graphverify_hybrid,llm_text_verifier \\
      --output output/results/runtime_benchmark.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.answer_generation import generate_answer, generate_answers_for_dataset
from experiments._common import add_llm_args, build_config, build_llm_client, load_samples, parse_dataset_list, save_csv
from experiments._methods import build_method
from graphverify.claim_decomposer import ClaimDecomposer


def _peak_memory_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return float("nan")


def benchmark_base_rag(samples: List[Dict[str, Any]], llm_client) -> Dict[str, Any]:
    """Times answer generation alone (no verification), the "base RAG pipeline" cost baseline."""
    start_mem = _peak_memory_mb()
    t0 = time.perf_counter()
    n = 0
    for sample in samples:
        generate_answer(llm_client, sample.get("query", ""), sample.get("passages", []))
        n += 1
    elapsed = time.perf_counter() - t0
    return {
        "sec_per_answer": elapsed / n if n else 0.0,
        "n_answers": n,
        "peak_memory_mb": max(start_mem, _peak_memory_mb()),
    }


def benchmark_method(method_name: str, samples: List[Dict[str, Any]], llm_client, cfg) -> Dict[str, Any]:
    decomposer = ClaimDecomposer(llm_client)
    method = build_method(method_name, llm_client, cfg)

    n_answers = n_claims = 0
    start_mem = _peak_memory_mb()
    t0 = time.perf_counter()
    for sample in samples:
        answer = sample.get("generated") or sample.get("answer", "")
        if not answer:
            continue
        claims = decomposer.decompose(answer)
        if not claims:
            continue
        method.verify(sample.get("query", ""), sample.get("passages", []), claims)
        n_answers += 1
        n_claims += len(claims)
    elapsed = time.perf_counter() - t0

    return {
        "sec_per_answer": elapsed / n_answers if n_answers else 0.0,
        "sec_per_claim": elapsed / n_claims if n_claims else 0.0,
        "n_answers": n_answers, "n_claims": n_claims,
        "peak_memory_mb": max(start_mem, _peak_memory_mb()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=50)
    add_llm_args(p)
    p.add_argument("--methods", type=str, default="graphverify_score,graphverify_hybrid,llm_text_verifier")
    p.add_argument("--output", type=str, default="output/results/runtime_benchmark.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    llm_client = build_llm_client(args)
    method_names = parse_dataset_list(args.methods)

    samples = load_samples(args.dataset, args)
    samples = generate_answers_for_dataset(llm_client, samples)

    base = benchmark_base_rag(samples, llm_client)
    print(f"base RAG pipeline: {base['sec_per_answer']:.3f} sec/answer, {base['peak_memory_mb']:.1f} MB RSS")

    rows = [{"method": "base_rag_generation_only", "dataset": args.dataset, **base, "relative_cost_vs_base": 1.0}]
    for method_name in method_names:
        m = benchmark_method(method_name, samples, llm_client, cfg)
        relative = (m["sec_per_answer"] / base["sec_per_answer"]) if base["sec_per_answer"] else float("nan")
        rows.append({"method": method_name, "dataset": args.dataset, **m, "relative_cost_vs_base": relative})
        print(f"{method_name}: {m['sec_per_answer']:.3f} sec/answer ({relative:.2f}x base), "
              f"{m['sec_per_claim']:.3f} sec/claim, {m['peak_memory_mb']:.1f} MB RSS")

    save_csv(rows, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
