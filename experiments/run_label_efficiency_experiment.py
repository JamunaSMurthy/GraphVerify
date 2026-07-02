"""
Label-efficiency experiment: directly defends GraphVerify-score's
"training-free" claim by showing how little labeled dev data its two
thresholds (support/contradiction) actually need.

GraphVerify-score has no trainable weights, so "training with X% of
labels" can only mean one thing here: selecting the two thresholds
(`support_threshold`, `contradict_threshold`) by grid search using only a
random X% subsample of the dev set's gold labels, then evaluating the
resulting fixed thresholds on the full test set. A "0%" condition uses the
untuned defaults (0.60 / 0.55) directly, with no dev-label access at all.

This is computationally cheap because path scores
(`support_score`/`contradict_score` on each claim's cached
VerificationRecord) do not depend on the threshold -- the pipeline runs
once per dev/test split, and every candidate threshold pair and every label
fraction is evaluated by re-deriving verdicts from the cached scores via
`graphverify.verdict_assigner.VerdictAssigner.verdict_from_scores`, with no
repeated LLM calls or path search.

Usage:
  python experiments/run_label_efficiency_experiment.py \\
      --dataset hotpotqa --split validation --max_samples 300 \\
      --dev_fraction 0.3 --seed 0 \\
      --output output/results/label_efficiency.csv
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.answer_generation import generate_answers_for_dataset
from dataset.splits import apply_split, build_split
from eval.metrics import claim_accuracy, compute_all_metrics
from experiments._common import add_dataset_args, add_llm_args, build_config, build_llm_client, load_samples, save_csv
from graphverify.claim_decomposer import ClaimDecomposer
from graphverify.entity_linker import EntityLinker
from graphverify.evidence_graph import EvidenceGraphBuilder
from graphverify.path_scorer import PathScorer
from graphverify.path_searcher import PathSearcher
from graphverify.relation_normalizer import RelationNormalizer
from graphverify.triple_extractor import TripleExtractor
from graphverify.verdict_assigner import VerdictAssigner

LABEL_FRACTIONS = (0.0, 0.01, 0.05, 0.10, 0.25, 1.00)
SUPPORT_GRID = tuple(round(0.4 + 0.05 * i, 2) for i in range(9))     # 0.40 .. 0.80
CONTRADICT_GRID = tuple(round(0.4 + 0.05 * i, 2) for i in range(9))  # 0.40 .. 0.80
DEFAULT_SUPPORT_THRESHOLD = 0.60
DEFAULT_CONTRADICT_THRESHOLD = 0.55


def compute_claim_scores(
    samples: List[Dict[str, Any]],
    llm_client,
    cfg,
) -> List[Dict[str, Any]]:
    """
    Runs claim decomposition, triple extraction, and path search/scoring
    once per sample (independent of any threshold), returning one record
    per claim with its raw `support_score`/`contradict_score` and gold
    label. This is the expensive step; grid search over thresholds and
    label fractions afterward touches none of this.
    """
    decomposer = ClaimDecomposer(llm_client)
    rel_norm = RelationNormalizer(embed_model=cfg.embed_model, cosine_cutoff=cfg.embed_cosine_cutoff)
    scorer = PathScorer(
        lambda_head=cfg.lambda_head, lambda_rel=cfg.lambda_rel, lambda_tail=cfg.lambda_tail,
        lambda_prov=cfg.lambda_prov, embed_model=cfg.embed_model, cosine_cutoff=cfg.embed_cosine_cutoff,
    )
    builder = EvidenceGraphBuilder(llm_client=llm_client, relation_normalizer=rel_norm, embed_model=cfg.embed_model)

    scored_claims: List[Dict[str, Any]] = []
    for sample in samples:
        answer = sample.get("generated") or sample.get("answer", "")
        if not answer:
            continue
        claims = decomposer.decompose(answer)
        if not claims:
            continue

        graph = builder.build(sample.get("query", ""), sample.get("passages", []))
        entity_linker = EntityLinker(graph.node_labels, embed_model=cfg.embed_model, cosine_cutoff=cfg.embed_cosine_cutoff)
        triple_extractor = TripleExtractor(llm_client, entity_linker, rel_norm)
        path_searcher = PathSearcher(graph=graph, entity_linker=entity_linker, path_scorer=scorer, l_max=cfg.l_max, top_k=cfg.top_k_paths)

        for i, claim in enumerate(claims):
            triple = triple_extractor.extract(claim)
            s_plus = s_minus = 0.0
            if triple.linked:
                support_paths, conflict_paths = path_searcher.search(head=triple.head, relation=triple.relation, tail=triple.tail)
                s_plus = max((p.score for p in support_paths), default=0.0)
                s_minus = max((p.score for p in conflict_paths), default=0.0)
            scored_claims.append({
                "item_id": f"{sample['id']}::{i}", "sample_id": str(sample["id"]),
                "support_score": s_plus, "contradict_score": s_minus,
                "gold_verdict": sample.get("gold_verdict", "Unsupported"),
            })
    return scored_claims


def select_thresholds(
    dev_claims: List[Dict[str, Any]],
    label_fraction: float,
    seed: int,
    support_grid=SUPPORT_GRID,
    contradict_grid=CONTRADICT_GRID,
) -> Tuple[float, float]:
    """Grid-searches (support_threshold, contradict_threshold) maximizing claim accuracy on a `label_fraction` subsample of `dev_claims`."""
    if label_fraction <= 0.0 or not dev_claims:
        return DEFAULT_SUPPORT_THRESHOLD, DEFAULT_CONTRADICT_THRESHOLD

    rng = random.Random(seed)
    n_labeled = max(1, round(len(dev_claims) * label_fraction))
    labeled = rng.sample(dev_claims, min(n_labeled, len(dev_claims)))

    best_acc, best_ts, best_tc = -1.0, DEFAULT_SUPPORT_THRESHOLD, DEFAULT_CONTRADICT_THRESHOLD
    for ts in support_grid:
        for tc in contradict_grid:
            assigner = VerdictAssigner(support_threshold=ts, contradict_threshold=tc)
            preds = [assigner.verdict_from_scores(c["support_score"], c["contradict_score"]) for c in labeled]
            golds = [c["gold_verdict"] for c in labeled]
            acc = claim_accuracy(preds, golds)
            if acc > best_acc:
                best_acc, best_ts, best_tc = acc, ts, tc
    return best_ts, best_tc


def evaluate_thresholds(test_claims: List[Dict[str, Any]], support_threshold: float, contradict_threshold: float) -> Dict[str, float]:
    assigner = VerdictAssigner(support_threshold=support_threshold, contradict_threshold=contradict_threshold)
    preds = [assigner.verdict_from_scores(c["support_score"], c["contradict_score"]) for c in test_claims]
    golds = [c["gold_verdict"] for c in test_claims]
    return compute_all_metrics(preds, golds)


def run_label_efficiency(
    dev_claims: List[Dict[str, Any]],
    test_claims: List[Dict[str, Any]],
    seed: int = 0,
    fractions=LABEL_FRACTIONS,
) -> List[Dict[str, Any]]:
    rows = []
    for fraction in fractions:
        ts, tc = select_thresholds(dev_claims, fraction, seed=seed)
        metrics = evaluate_thresholds(test_claims, ts, tc)
        rows.append({"label_fraction": fraction, "support_threshold": ts, "contradict_threshold": tc, **metrics})
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(p, default_datasets="hotpotqa")
    add_llm_args(p)
    p.add_argument("--dataset", type=str, default="hotpotqa")
    p.add_argument("--dev_fraction", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default="output/results/label_efficiency.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    llm_client = build_llm_client(args)

    samples = load_samples(args.dataset, args)
    samples = generate_answers_for_dataset(llm_client, samples)

    split = build_split([str(s["id"]) for s in samples], args.dataset, dev_fraction=args.dev_fraction, seed=args.seed)
    dev_samples = apply_split(samples, split, "dev")
    test_samples = apply_split(samples, split, "test")

    print(f"Scoring {len(dev_samples)} dev + {len(test_samples)} test samples...")
    dev_claims = compute_claim_scores(dev_samples, llm_client, cfg)
    test_claims = compute_claim_scores(test_samples, llm_client, cfg)

    rows = run_label_efficiency(dev_claims, test_claims, seed=args.seed)

    print(f"\n{'Label %':>8} {'theta_s':>8} {'theta_c':>8} {'ClaimAcc':>9} {'MacroF1':>8}")
    for r in rows:
        print(f"{r['label_fraction'] * 100:>7.0f}% {r['support_threshold']:>8.2f} {r['contradict_threshold']:>8.2f} "
              f"{r.get('claim_acc', 0):>9.1f} {r.get('macro_f1', 0):>8.1f}")

    save_csv(rows, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
