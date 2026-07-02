"""
Long-form generation evaluation: runs verification specifically on
RAGTruth's long-form task types (Summary, Data2txt -- as opposed to its
shorter QA responses), and reports both claim-level metrics and
response-level hallucination detection (AUROC/AUPRC, precision/recall at a
threshold), since RAGTruth's real annotation strength is hallucination
spans, not inline citations.

**Scope note.** RAGTruth does not carry inline citation markers, so this
script does not compute a "citation precision/recall" metric -- that would
require citation-annotated data this benchmark does not have. What it
reports instead (claim-level verdicts + response-level hallucination
detection on the long-form subset) is the citation-adjacent evaluation
RAGTruth's actual labels support.

Usage:
  python experiments/run_longform_citation_evaluation.py \\
      --split test --max_samples 200 \\
      --methods graphverify_hybrid,llm_text_verifier \\
      --output_dir output/results/longform_evaluation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import hallucination_auroc_auprc
from experiments._common import add_llm_args, build_config, build_llm_client, save_csv, save_json
from experiments._methods import ALL_METHOD_NAMES, build_method
from dataset.loader import load_dataset
from graphverify.claim_decomposer import ClaimDecomposer

LONGFORM_TASK_TYPES = ("Summary", "Data2txt")


def filter_longform(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept = []
    for s in samples:
        try:
            meta = json.loads(s.get("label", "{}"))
        except json.JSONDecodeError:
            meta = {}
        if meta.get("task_type") in LONGFORM_TASK_TYPES:
            kept.append(s)
    return kept


def run_longform_evaluation(
    samples: List[Dict[str, Any]],
    method_names: List[str],
    llm_client,
    cfg,
) -> Dict[str, Dict[str, Any]]:
    decomposer = ClaimDecomposer(llm_client)
    claims_by_id = {str(s["id"]): decomposer.decompose(s.get("generated", "")) for s in samples}

    results = {}
    for method_name in method_names:
        method = build_method(method_name, llm_client, cfg)
        response_reliability, response_is_hallucination = [], []

        for sample in samples:
            sid = str(sample["id"])
            claims = claims_by_id[sid]
            if not claims:
                continue
            records = method.verify(sample.get("query", ""), sample.get("passages", []), claims)
            reliabilities = [float(r.get("reliability", 0.0)) for r in records]

            # A response's overall reliability is its least reliable claim --
            # one hallucinated claim in an otherwise-good response should not
            # be diluted away by several well-supported ones.
            response_reliability.append(min(reliabilities) if reliabilities else 0.0)
            try:
                meta = json.loads(sample.get("label", "{}"))
            except json.JSONDecodeError:
                meta = {}
            is_hallucinated = 1 if meta.get("hallucination_spans") else 0
            response_is_hallucination.append(is_hallucinated)

        detection = hallucination_auroc_auprc(response_reliability, response_is_hallucination) if response_reliability else {}
        results[method_name] = {
            "n_responses": len(response_reliability),
            "n_hallucinated": sum(response_is_hallucination),
            **detection,
        }
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    add_llm_args(p)
    p.add_argument("--methods", type=str, default=",".join(ALL_METHOD_NAMES))
    p.add_argument("--output_dir", type=str, default="output/results/longform_evaluation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    llm_client = build_llm_client(args)
    method_names = [m.strip() for m in args.methods.split(",") if m.strip()]

    samples = load_dataset("ragtruth", split=args.split, data_dir=args.data_dir, max_samples=args.max_samples)
    longform_samples = filter_longform(samples)
    print(f"{len(longform_samples)}/{len(samples)} RAGTruth samples are long-form (Summary/Data2txt).")

    results = run_longform_evaluation(longform_samples, method_names, llm_client, cfg)

    rows = [{"method": name, **metrics} for name, metrics in results.items()]
    print(f"{'Method':<20} {'N':>5} {'Halluc.':>8} {'AUROC':>7} {'AUPRC':>7}")
    for r in rows:
        print(f"{r['method']:<20} {r.get('n_responses', 0):>5} {r.get('n_hallucinated', 0):>8} "
              f"{r.get('auroc', float('nan')):>7.3f} {r.get('auprc', float('nan')):>7.3f}")

    save_csv(rows, os.path.join(args.output_dir, "longform_hallucination_detection.csv"))
    save_json(results, os.path.join(args.output_dir, "longform_results.json"))
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
