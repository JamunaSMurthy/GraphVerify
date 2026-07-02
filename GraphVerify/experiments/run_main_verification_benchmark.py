"""
Runs the main claim-level verification benchmark: GraphVerify-score,
GraphVerify-hybrid, and all nine baselines, on the same generated answers,
retrieved passages, and decomposed claims, across one or more datasets.
Produces claim accuracy, per-class F1, macro-F1, path correctness, and ECE
per method, with 95% paired-bootstrap confidence intervals clustered by
answer id.

This script deliberately does not report "mean +/- std over N verifier
seeds": every LLM call in this pipeline uses greedy/deterministic decoding
(``GraphVerifyConfig.llm_temperature == 0.0`` by default), and the
score-only path-scoring/verdict-assignment logic is a fixed rule, so
repeating a run with a different seed measures no real source of variance
-- it would just reproduce the same output at additional API cost.
Uncertainty is instead reported via paired bootstrap confidence intervals
over claims, clustered by the answer/question each claim was decomposed
from (claims from one answer are not independent -- a single bad retrieval
affects every claim drawn from it).

**Claim-level gold labels.** The unified dataset schema
(``dataset/loader.py``) carries one ``gold_verdict`` per *answer*, not per
decomposed claim. By default this script broadcasts that answer-level label
to every claim decomposed from it -- an honest but coarse proxy, since a
single answer can contain a mix of supported and unsupported claims even
when its overall gold label is one value. Passing ``--annotations_dir``
with real per-claim adjudicated labels (``dataset/claim_annotation.py``
schema, keyed by item_id ``"{sample_id}::{claim_index}"``) overrides the
proxy label wherever a real annotation exists. Any results table produced
without ``--annotations_dir`` should be captioned as using proxy labels.

Usage:
  python experiments/run_main_verification_benchmark.py \\
      --datasets hotpotqa,2wikimultihopqa,musique,fever,ragtruth \\
      --split validation --max_samples 200 \\
      --methods graphverify_score,graphverify_hybrid,safe,rarr,fire,citefix,graphrag_adapted,hipporag_adapted,graphcheck_adapted,hybrid_kg_llm,llm_text_verifier \\
      --output_dir output/results/main_benchmark
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.answer_generation import generate_answers_for_dataset
from dataset.claim_annotation import load_annotation_csv
from eval.metrics import compute_all_metrics, run_bootstrap
from experiments._common import (
    add_dataset_args, add_llm_args, build_config, build_llm_client,
    load_samples, parse_dataset_list, save_csv, save_json,
)
from experiments._methods import ALL_METHOD_NAMES, build_method
from graphverify.claim_decomposer import ClaimDecomposer


def load_claim_level_annotations(annotations_dir: Optional[str]) -> Dict[str, str]:
    """Returns {item_id: verdict} from every annotation CSV in `annotations_dir` (empty dict if None)."""
    if not annotations_dir:
        return {}
    overrides: Dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(annotations_dir, "*.csv"))):
        for rec in load_annotation_csv(path):
            overrides[rec.item_id] = rec.verdict
    return overrides


def run_benchmark_on_dataset(
    dataset: str,
    samples: List[Dict[str, Any]],
    method_names: List[str],
    llm_client,
    cfg,
    annotation_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Runs every method in `method_names` over `samples` (already carrying a
    non-empty ``"generated"`` field). Returns
    ``{method_name: {"predictions": [...], "metrics": {...}}}``.
    """
    annotation_overrides = annotation_overrides or {}
    decomposer = ClaimDecomposer(llm_client)

    claims_by_id: Dict[str, List[str]] = {}
    for sample in samples:
        sid = str(sample.get("id", ""))
        claims_by_id[sid] = decomposer.decompose(sample.get("generated", ""))

    results: Dict[str, Dict[str, Any]] = {}
    for method_name in method_names:
        method = build_method(method_name, llm_client, cfg)
        predictions: List[Dict[str, Any]] = []
        preds, golds, pred_paths, gold_paths, rel_scores, cluster_ids = [], [], [], [], [], []

        for sample in samples:
            sid = str(sample.get("id", ""))
            claims = claims_by_id[sid]
            if not claims:
                continue
            records = method.verify(sample.get("query", ""), sample.get("passages", []), claims)
            predictions.append({
                "id": sid, "dataset": dataset, "method": method_name,
                "query": sample.get("query", ""), "generated": sample.get("generated", ""),
                "gold_verdict": sample.get("gold_verdict", ""), "records": records,
            })
            for i, rec in enumerate(records):
                item_id = f"{sid}::{i}"
                gold = annotation_overrides.get(item_id, sample.get("gold_verdict", "Unsupported"))
                preds.append(rec.get("verdict", "Unsupported"))
                golds.append(gold)
                pred_paths.append(_path_to_str(rec.get("best_path")))
                gold_paths.append(sample.get("gold_path", ""))
                rel_scores.append(float(rec.get("reliability", 0.0)))
                cluster_ids.append(sid)

        metrics = compute_all_metrics(preds, golds, pred_paths=pred_paths, gold_paths=gold_paths, rel_scores=rel_scores)
        if preds:
            for metric_name in ("claim_acc", "macro_f1", "unsupp_f1", "contr_f1"):
                point, lo, hi = run_bootstrap(preds, golds, metric=metric_name, cluster_ids=cluster_ids, n_boot=1000)
                metrics[f"{metric_name}_ci_low"] = lo
                metrics[f"{metric_name}_ci_high"] = hi
        metrics["n_claims"] = len(preds)

        results[method_name] = {"predictions": predictions, "metrics": metrics}
    return results


def _path_to_str(path) -> str:
    if not path:
        return ""
    if isinstance(path, str):
        return path
    if isinstance(path, list):
        parts = []
        for e in path:
            if isinstance(e, dict):
                parts.append(f"{e.get('src_label', e.get('src', ''))} -> {e.get('relation', '')} -> {e.get('dst_label', e.get('dst', ''))}")
            else:
                parts.append(str(e))
        return "; ".join(parts)
    return ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(p)
    add_llm_args(p)
    p.add_argument("--methods", type=str, default=",".join(ALL_METHOD_NAMES))
    p.add_argument("--evidence_mode", type=str, default="hybrid", choices=["text", "retrieved_graph", "kg_paths", "hybrid"])
    p.add_argument("--generated_answers", type=str, default=None, help="JSONL of {id, generated_answer} to reuse instead of generating fresh answers.")
    p.add_argument("--annotations_dir", type=str, default=None, help="Directory of claim-level annotation CSVs overriding the answer-level proxy label.")
    p.add_argument("--output_dir", type=str, default="output/results/main_benchmark")
    return p.parse_args()


def _load_generated_cache(path: Optional[str]) -> Dict[str, str]:
    if not path or not os.path.exists(path):
        return {}
    cache = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                cache[str(rec["id"])] = rec.get("generated_answer", rec.get("answer", ""))
    return cache


def main() -> None:
    args = parse_args()
    datasets = parse_dataset_list(args.datasets)
    method_names = parse_dataset_list(args.methods)
    cfg = build_config(args, evidence_mode=args.evidence_mode)
    llm_client = build_llm_client(args)
    annotation_overrides = load_claim_level_annotations(args.annotations_dir)
    generated_cache = _load_generated_cache(args.generated_answers)

    print(f"{'Dataset':<16} {'Method':<20} {'ClaimAcc':>9} {'MacroF1':>8} {'UnsuppF1':>9} {'ContrF1':>8} {'PathCorr':>9} {'ECE':>7}")
    print("-" * 90)

    for dataset in datasets:
        samples = load_samples(dataset, args)
        for sample in samples:
            sid = str(sample.get("id", ""))
            if sid in generated_cache:
                sample["generated"] = generated_cache[sid]
        samples = generate_answers_for_dataset(llm_client, samples)

        dataset_results = run_benchmark_on_dataset(dataset, samples, method_names, llm_client, cfg, annotation_overrides)

        for method_name, result in dataset_results.items():
            m = result["metrics"]
            print(f"{dataset:<16} {method_name:<20} {m.get('claim_acc', 0):>9.1f} {m.get('macro_f1', 0):>8.1f} "
                  f"{m.get('unsupp_f1', 0):>9.1f} {m.get('contr_f1', 0):>8.1f} {m.get('path_corr', 0):>9.1f} {m.get('ece', 0):>7.3f}")

            out_dir = os.path.join(args.output_dir, dataset)
            save_json(result["predictions"], os.path.join(out_dir, f"{method_name}_predictions.json"))

        summary_rows = [
            {"dataset": dataset, "method": name, **result["metrics"]}
            for name, result in dataset_results.items()
        ]
        save_csv(summary_rows, os.path.join(args.output_dir, dataset, "metrics_summary.csv"))

    print(f"\nSaved predictions and metrics under {args.output_dir}")


if __name__ == "__main__":
    main()
