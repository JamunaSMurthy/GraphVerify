"""
Evaluation metrics for claim-level verification:
  Claim Accuracy, per-class F1 (Supported / Unsupported / Contradictory),
  macro-F1, Path Correctness, ECE (aggregate and per-class), AUROC/AUPRC for
  hallucination detection, and paired bootstrap confidence intervals
  (optionally clustered by answer/question).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


LABEL2IDX = {
    "Supported":     0,
    "Unsupported":   1,
    "Contradictory": 2,
}
VERDICT_CLASSES = ("Supported", "Unsupported", "Contradictory")


def claim_accuracy(preds: List[str], golds: List[str]) -> float:
    """Three-way verdict accuracy (Supported / Unsupported / Contradictory)."""
    assert len(preds) == len(golds)
    return float(accuracy_score(golds, preds) * 100)


def supported_f1(preds: List[str], golds: List[str]) -> float:
    """F1 for the Supported class."""
    return _class_f1(preds, golds, "Supported")


def unsupported_f1(preds: List[str], golds: List[str]) -> float:
    """F1 for the Unsupported class."""
    return _class_f1(preds, golds, "Unsupported")


def contradiction_f1(preds: List[str], golds: List[str]) -> float:
    """F1 for the Contradictory class."""
    return _class_f1(preds, golds, "Contradictory")


def macro_f1(preds: List[str], golds: List[str]) -> float:
    """
    Macro-averaged F1 across all three verdict classes: the mean of
    Supported F1, Unsupported F1, and Contradiction F1, each computed
    against its own binary one-vs-rest split. Unlike claim accuracy, this
    is not dominated by the majority class, and unlike a single class's F1
    it does not ignore the other two — the revision plan requires reporting
    this "in addition to individual Unsupported/Contradiction F1" because
    contradiction prevalence is typically low and a high aggregate accuracy
    can hide a near-zero contradiction F1.
    """
    scores = [_class_f1(preds, golds, cls) for cls in VERDICT_CLASSES]
    return float(np.mean(scores))


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


def per_class_ece(
    preds: List[str],
    golds: List[str],
    scores: List[float],
    n_bins: int = 15,
) -> Dict[str, float]:
    """
    ECE computed separately for each predicted verdict class, plus an
    "overall" entry equal to :func:`expected_calibration_error` over all
    claims. The revision plan requires ECE broken out by verdict class
    because a verifier can be well-calibrated in aggregate while being
    systematically over- or under-confident on one class (most often
    Contradictory, the rarest and highest-stakes class) — an aggregate-only
    ECE would hide that.
    """
    assert len(preds) == len(golds) == len(scores)
    correct = [1 if p == g else 0 for p, g in zip(preds, golds)]
    result = {"overall": expected_calibration_error(scores, correct, n_bins=n_bins)}
    for cls in VERDICT_CLASSES:
        idx = [i for i, p in enumerate(preds) if p == cls]
        if not idx:
            continue
        result[cls] = expected_calibration_error(
            [scores[i] for i in idx], [correct[i] for i in idx], n_bins=n_bins,
        )
    return result


def hallucination_auroc_auprc(scores: List[float], is_hallucination: Sequence[int]) -> Dict[str, float]:
    """
    AUROC and AUPRC for hallucination detection: `scores` is the verifier's
    reliability/confidence that a *response* is faithful (higher = more
    reliable), `is_hallucination` is 1 if the response actually contains a
    hallucination (e.g. RAGTruth's response-level label) and 0 otherwise.
    Detection score used is ``1 - reliability`` so higher values indicate
    higher hallucination risk, matching the sign convention `roc_auc_score`/
    `average_precision_score` expect for the positive (hallucination) class.
    Returns {"auroc": ..., "auprc": ...}; both are NaN if `is_hallucination`
    is all-0 or all-1 (undefined without both classes present).
    """
    assert len(scores) == len(is_hallucination)
    labels = np.asarray(is_hallucination)
    if labels.min() == labels.max():
        return {"auroc": float("nan"), "auprc": float("nan")}
    risk_scores = 1.0 - np.asarray(scores, dtype=np.float64)
    return {
        "auroc": float(roc_auc_score(labels, risk_scores)),
        "auprc": float(average_precision_score(labels, risk_scores)),
    }


def compute_all_metrics(
    preds:      List[str],
    golds:      List[str],
    pred_paths: Optional[List[Optional[str]]] = None,
    gold_paths: Optional[List[Optional[str]]] = None,
    rel_scores: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute the core claim-level metrics at once. Returns a dict keyed by metric name."""
    results: Dict[str, float] = {
        "claim_acc": claim_accuracy(preds, golds),
        "supp_f1":   supported_f1(preds, golds),
        "unsupp_f1": unsupported_f1(preds, golds),
        "contr_f1":  contradiction_f1(preds, golds),
        "macro_f1":  macro_f1(preds, golds),
    }
    if pred_paths is not None and gold_paths is not None:
        results["path_corr"] = path_correctness(pred_paths, gold_paths)
    if rel_scores is not None:
        correct = [1 if p == g else 0 for p, g in zip(preds, golds)]
        results["ece"] = expected_calibration_error(rel_scores, correct)
    return results


BOOTSTRAP_METRICS = {
    "claim_acc": claim_accuracy,
    "supp_f1":   supported_f1,
    "unsupp_f1": unsupported_f1,
    "contr_f1":  contradiction_f1,
    "macro_f1":  macro_f1,
}


def run_bootstrap(
    preds: List[str],
    golds: List[str],
    n_boot: int = 1000,
    metric: str = "claim_acc",
    alpha: float = 0.05,
    seed: int = 42,
    cluster_ids: Optional[List[str]] = None,
) -> Tuple[float, float, float]:
    """
    Paired bootstrap confidence interval over claim-level predictions.
    Returns (point_estimate, lower_ci, upper_ci).

    If `cluster_ids` is given (one id per claim, e.g. the answer or question
    id each claim was decomposed from), resampling is done at the cluster
    level: each bootstrap replicate samples cluster ids with replacement and
    takes *all* claims belonging to each sampled cluster, rather than
    resampling individual claims independently. This is required whenever
    claims from the same answer are not exchangeable (verdict errors are
    often correlated within an answer — e.g. a single bad retrieval affects
    every claim drawn from it), per the revision plan's statistics section
    ("account for clustering by answer/question"). Without `cluster_ids`,
    plain i.i.d. claim-level resampling is used, which understates variance
    whenever such correlation exists.
    """
    rng = np.random.default_rng(seed)
    n = len(preds)
    metric_fn = BOOTSTRAP_METRICS[metric]

    point = metric_fn(preds, golds)
    boots = []

    if cluster_ids is not None:
        assert len(cluster_ids) == n
        by_cluster: Dict[str, List[int]] = defaultdict(list)
        for i, cid in enumerate(cluster_ids):
            by_cluster[cid].append(i)
        clusters = list(by_cluster.keys())
        n_clusters = len(clusters)
        for _ in range(n_boot):
            sampled_clusters = rng.choice(clusters, size=n_clusters, replace=True)
            idx = [i for cid in sampled_clusters for i in by_cluster[cid]]
            boots.append(metric_fn([preds[i] for i in idx], [golds[i] for i in idx]))
    else:
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
