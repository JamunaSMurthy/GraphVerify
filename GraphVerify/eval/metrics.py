"""
Evaluation metrics for claim-level verification:
  Claim Accuracy, Unsupported F1, Contradiction F1, Path Correctness, ECE.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


LABEL2IDX = {
    "Supported":     0,
    "Unsupported":   1,
    "Contradictory": 2,
}


def claim_accuracy(preds: List[str], golds: List[str]) -> float:
    """Three-way verdict accuracy (Supported / Unsupported / Contradictory)."""
    assert len(preds) == len(golds)
    return float(accuracy_score(golds, preds) * 100)


def unsupported_f1(preds: List[str], golds: List[str]) -> float:
    """F1 for the Unsupported class."""
    return _class_f1(preds, golds, "Unsupported")


def contradiction_f1(preds: List[str], golds: List[str]) -> float:
    """F1 for the Contradictory class."""
    return _class_f1(preds, golds, "Contradictory")


def path_correctness(
    pred_paths: List[Optional[str]],
    gold_paths: List[Optional[str]],
    method: str = "f1_token",
) -> float:
    """
    Percentage of returned evidence paths that match gold paths.
    method: "exact" for exact string match, "f1_token" for token-level F1.
    """
    assert len(pred_paths) == len(gold_paths)
    scores = []
    for pred, gold in zip(pred_paths, gold_paths):
        if gold is None or gold == "":
            if not pred:
                scores.append(1.0)
            continue
        if not pred:
            scores.append(0.0)
            continue
        if method == "exact":
            scores.append(1.0 if _norm(pred) == _norm(gold) else 0.0)
        else:
            scores.append(_token_f1(pred, gold))
    return float(np.mean(scores) * 100) if scores else 0.0


def expected_calibration_error(
    scores: List[float],
    labels: List[int],
    n_bins: int = 15,
) -> float:
    """ECE: labels 1 = prediction correct, 0 = incorrect."""
    from graphverify.calibrator import compute_ece
    return compute_ece(scores, labels, n_bins=n_bins)


def compute_all_metrics(
    preds:      List[str],
    golds:      List[str],
    pred_paths: Optional[List[Optional[str]]] = None,
    gold_paths: Optional[List[Optional[str]]] = None,
    rel_scores: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute all five metrics at once. Returns a dict keyed by metric name."""
    results: Dict[str, float] = {
        "claim_acc": claim_accuracy(preds, golds),
        "unsupp_f1": unsupported_f1(preds, golds),
        "contr_f1":  contradiction_f1(preds, golds),
    }
    if pred_paths is not None and gold_paths is not None:
        results["path_corr"] = path_correctness(pred_paths, gold_paths)
    if rel_scores is not None:
        correct = [1 if p == g else 0 for p, g in zip(preds, golds)]
        results["ece"] = expected_calibration_error(rel_scores, correct)
    return results


def run_bootstrap(
    preds: List[str],
    golds: List[str],
    n_boot: int = 1000,
    metric: str = "claim_acc",
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Paired bootstrap confidence interval.
    Returns (point_estimate, lower_ci, upper_ci).
    """
    rng = np.random.default_rng(seed)
    n = len(preds)
    metric_fn = {
        "claim_acc": claim_accuracy,
        "unsupp_f1": unsupported_f1,
        "contr_f1":  contradiction_f1,
    }[metric]

    point = metric_fn(preds, golds)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(metric_fn([preds[i] for i in idx], [golds[i] for i in idx]))

    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return point, lo, hi


def _class_f1(preds: List[str], golds: List[str], target_class: str) -> float:
    p_bin = [1 if x == target_class else 0 for x in preds]
    g_bin = [1 if x == target_class else 0 for x in golds]
    if sum(g_bin) == 0:
        return 0.0
    _, _, f1, _ = precision_recall_fscore_support(g_bin, p_bin, average="binary", zero_division=0)
    return float(f1 * 100)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip().replace("→", ">"))


def _token_f1(pred: str, gold: str) -> float:
    pred_toks = _norm(pred).split()
    gold_toks = _norm(gold).split()
    common    = Counter(pred_toks) & Counter(gold_toks)
    n_common  = sum(common.values())
    if n_common == 0:
        return 0.0
    prec = n_common / len(pred_toks)
    rec  = n_common / len(gold_toks)
    return 2 * prec * rec / (prec + rec)
