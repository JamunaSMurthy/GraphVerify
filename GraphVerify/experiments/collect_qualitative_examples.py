"""
Collects a small, compact set of qualitative examples (real predictions,
not fabricated) spanning Supported/Unsupported/Contradictory verdicts,
formatted with claim, triple, verdict, and evidence-path trace -- exactly
the shape needed for a qualitative-examples table.

Reads prediction files produced by `experiments/run_main_verification_benchmark.py`;
samples deterministically (seeded) rather than taking the first N, since
the first N predictions in file order are not a representative or
cherry-pick-resistant sample.

Usage:
  python experiments/collect_qualitative_examples.py \\
      --predictions_dir output/results/main_benchmark \\
      --dataset hotpotqa --method graphverify_hybrid \\
      --n_per_verdict 2 --seed 0 \\
      --output output/results/qualitative_examples.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_predictions(predictions_dir: str, dataset: str, method: str) -> List[Dict[str, Any]]:
    path = os.path.join(predictions_dir, dataset, f"{method}_predictions.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No predictions found at {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect_examples(
    samples: List[Dict[str, Any]],
    n_per_verdict: int = 2,
    seed: int = 0,
    require_path: bool = True,
) -> List[Dict[str, Any]]:
    """
    Samples up to `n_per_verdict` claim records per verdict class
    (Supported, Unsupported, Contradictory), deterministically. If
    `require_path` is True, Supported/Contradictory candidates without a
    non-empty `best_path` are excluded (a qualitative example with no
    evidence trace to show is not useful in a qualitative table).
    """
    by_verdict: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        for rec in sample.get("records", []):
            verdict = rec.get("verdict", "")
            if require_path and verdict in ("Supported", "Contradictory") and not rec.get("best_path"):
                continue
            by_verdict[verdict].append({
                "dataset_item_id": sample.get("id", ""),
                "query": sample.get("query", ""),
                "claim": rec.get("claim", ""),
                "triple": (rec.get("head"), rec.get("relation"), rec.get("tail")),
                "verdict": verdict,
                "reliability": rec.get("reliability", 0.0),
                "evidence_path": _format_path(rec.get("best_path")),
                "rationale": rec.get("rationale", ""),
            })

    rng = random.Random(seed)
    examples = []
    for verdict in ("Supported", "Unsupported", "Contradictory"):
        candidates = by_verdict.get(verdict, [])
        examples.extend(rng.sample(candidates, min(n_per_verdict, len(candidates))))
    return examples


def _format_path(path) -> str:
    if not path:
        return ""
    if isinstance(path, str):
        return path
    parts = []
    for e in path:
        if isinstance(e, dict):
            parts.append(f"{e.get('src_label', e.get('src', ''))} -> {e.get('relation', '')} -> {e.get('dst_label', e.get('dst', ''))}")
        else:
            parts.append(str(e))
    return "; ".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions_dir", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--method", type=str, required=True)
    p.add_argument("--n_per_verdict", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default="output/results/qualitative_examples.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_predictions(args.predictions_dir, args.dataset, args.method)
    examples = collect_examples(samples, n_per_verdict=args.n_per_verdict, seed=args.seed)

    for ex in examples:
        print(f"[{ex['verdict']}] {ex['claim']}")
        print(f"    triple: {ex['triple']}")
        print(f"    evidence: {ex['evidence_path'] or '(none)'}")
        print()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)
    print(f"Saved {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
