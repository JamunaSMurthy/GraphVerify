"""
Answer-level reliability: wires `eval/coverage_reliability.py`'s
accept/reject policy over a reliability-threshold sweep, reporting
accepted/rejected answer percentages, EM/F1 on the accepted subset, and
full-set EM/F1 -- the coverage/abstention table that separates a genuine
factuality gain from selection bias (accepting only the answers the
verifier finds easy).

Reports EM/F1 against the dataset's gold QA answer for QA-style datasets,
and RAGTruth-style hallucination precision/recall/AUROC/AUPRC (via
`eval.metrics.hallucination_auroc_auprc` and
`eval.coverage_reliability.hallucination_precision_recall`) when the
dataset is RAGTruth.

Usage:
  python experiments/run_answer_level_reliability.py \\
      --dataset hotpotqa --split validation --max_samples 200 \\
      --method graphverify_hybrid \\
      --thresholds 0.5,0.6,0.7,0.8,0.9 \\
      --output output/results/answer_level_reliability.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.answer_generation import generate_answers_for_dataset
from eval.coverage_reliability import coverage_reliability_report, hallucination_precision_recall
from eval.metrics import hallucination_auroc_auprc
from experiments._common import add_dataset_args, add_llm_args, build_config, build_llm_client, load_samples, save_csv
from experiments._methods import build_method
from graphverify.claim_decomposer import ClaimDecomposer


def build_answer_records(
    dataset: str,
    samples: List[Dict[str, Any]],
    method_name: str,
    llm_client,
    cfg,
) -> List[Dict[str, Any]]:
    decomposer = ClaimDecomposer(llm_client)
    method = build_method(method_name, llm_client, cfg)

    answers = []
    for sample in samples:
        answer = sample.get("generated") or sample.get("answer", "")
        if not answer:
            continue
        claims = decomposer.decompose(answer)
        if not claims:
            continue
        records = method.verify(sample.get("query", ""), sample.get("passages", []), claims)
        gold_answer = sample.get("answer", "") if dataset != "ragtruth" else ""
        answers.append({
            "id": sample["id"], "generated": answer, "gold_answer": gold_answer,
            "claim_records": records, "label": sample.get("label", ""),
        })
    return answers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(p, default_datasets="hotpotqa")
    p.add_argument("--dataset", type=str, default="hotpotqa")
    add_llm_args(p)
    p.add_argument("--method", type=str, default="graphverify_hybrid")
    p.add_argument("--thresholds", type=str, default="0.50,0.60,0.70,0.80,0.90")
    p.add_argument("--output", type=str, default="output/results/answer_level_reliability.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    llm_client = build_llm_client(args)
    thresholds = [float(x) for x in args.thresholds.split(",")]

    samples = load_samples(args.dataset, args)
    samples = generate_answers_for_dataset(llm_client, samples)
    answers = build_answer_records(args.dataset, samples, args.method, llm_client, cfg)

    coverage_results = coverage_reliability_report(answers, thresholds=thresholds)
    rows = [{
        "dataset": args.dataset, "method": args.method, "threshold": r.threshold,
        "accepted_pct": r.accepted_pct, "rejected_pct": r.rejected_pct,
        "accepted_em": r.accepted_em, "accepted_f1": r.accepted_f1,
        "full_set_em": r.full_set_em, "full_set_f1": r.full_set_f1,
        "n_answers": r.n_answers, "n_accepted": r.n_accepted,
    } for r in coverage_results]

    print(f"{'Threshold':>10} {'Accept%':>8} {'Reject%':>8} {'AccEM':>7} {'AccF1':>7} {'FullEM':>7} {'FullF1':>7}")
    for r in rows:
        print(f"{r['threshold']:>10.2f} {r['accepted_pct']:>8.1f} {r['rejected_pct']:>8.1f} "
              f"{r['accepted_em']:>7.1f} {r['accepted_f1']:>7.1f} {r['full_set_em']:>7.1f} {r['full_set_f1']:>7.1f}")

    if args.dataset == "ragtruth":
        scores, is_halluc = [], []
        for a in answers:
            reliabilities = [float(r.get("reliability", 0.0)) for r in a["claim_records"]]
            scores.append(min(reliabilities) if reliabilities else 0.0)
            try:
                meta = json.loads(a.get("label", "{}"))
            except json.JSONDecodeError:
                meta = {}
            is_halluc.append(1 if meta.get("hallucination_spans") else 0)

        detection = hallucination_auroc_auprc(scores, is_halluc)
        print(f"\nHallucination detection: AUROC={detection.get('auroc', float('nan')):.3f} AUPRC={detection.get('auprc', float('nan')):.3f}")
        for threshold in thresholds:
            pr = hallucination_precision_recall(scores, is_halluc, threshold)
            print(f"  threshold={threshold:.2f}: precision={pr['precision']:.1f} recall={pr['recall']:.1f} f1={pr['f1']:.1f}")

    save_csv(rows, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
