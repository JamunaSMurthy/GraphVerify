"""
Paired bootstrap significance testing between two systems evaluated on the
same claim set, with effect size reported alongside the p-value.

The revision plan's statistics section is explicit that this repository
must: use paired bootstrap over examples or claims (accounting for
clustering by answer/question when relevant), report significance only for
primary hypotheses, and report effect size rather than only "p < 0.05".
This module implements that; ``experiments/`` scripts call it once per
primary comparison (GraphVerify vs. the strongest baseline) rather than
running it pairwise across every method, to avoid the multiple-comparisons
problem the revision plan flags.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .metrics import BOOTSTRAP_METRICS


@dataclass
class SignificanceResult:
    metric:          str
    system_a_score:  float
    system_b_score:  float
    effect_size:     float   # system_a_score - system_b_score, in the metric's own units (points)
    p_value:         float   # two-sided, from the bootstrap distribution of the paired difference
    ci_low:          float
    ci_high:         float
    n_boot:          int


def paired_bootstrap_significance(
    preds_a: List[str],
    preds_b: List[str],
    golds: List[str],
    metric: str = "claim_acc",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    cluster_ids: Optional[List[str]] = None,
) -> SignificanceResult:
    """
    Tests whether system A's `metric` differs from system B's on the same
    underlying claims. `preds_a`, `preds_b`, and `golds` must be aligned
    claim-by-claim (same length and order — callers should join both
    systems' predictions on a shared claim/item id before calling this).

    Both systems are resampled with the *same* bootstrap indices/clusters
    on each replicate, which is what makes this a paired test: it isolates
    the variance of the *difference* rather than treating the two systems'
    scores as independent, so the resulting interval is tighter and the
    p-value more informative than an unpaired comparison would give.

    If `cluster_ids` is given (e.g. one id per claim identifying which
    answer/question it came from), resampling is done at the cluster level
    — see :func:`eval.metrics.run_bootstrap` for why this matters whenever
    claims from the same answer are not exchangeable.

    Returns effect size (`system_a_score - system_b_score`, in metric
    points) and a two-sided p-value computed as twice the smaller tail
    probability of the bootstrap difference distribution crossing zero.
    """
    assert len(preds_a) == len(preds_b) == len(golds)
    metric_fn = BOOTSTRAP_METRICS[metric]
    n = len(golds)

    rng = np.random.default_rng(seed)
    score_a = metric_fn(preds_a, golds)
    score_b = metric_fn(preds_b, golds)
    observed_diff = score_a - score_b

    if cluster_ids is not None:
        assert len(cluster_ids) == n
        by_cluster: Dict[str, List[int]] = {}
        for i, cid in enumerate(cluster_ids):
            by_cluster.setdefault(cid, []).append(i)
        clusters = list(by_cluster.keys())
        n_clusters = len(clusters)

        def sample_indices() -> List[int]:
            sampled = rng.choice(clusters, size=n_clusters, replace=True)
            return [i for cid in sampled for i in by_cluster[cid]]
    else:
        def sample_indices() -> List[int]:
            return list(rng.integers(0, n, size=n))

    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = sample_indices()
        a_score = metric_fn([preds_a[i] for i in idx], [golds[i] for i in idx])
        b_score = metric_fn([preds_b[i] for i in idx], [golds[i] for i in idx])
        diffs[b] = a_score - b_score

    ci_low = float(np.percentile(diffs, 100 * alpha / 2))
    ci_high = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    p_value = float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))

    return SignificanceResult(
        metric=metric, system_a_score=score_a, system_b_score=score_b,
        effect_size=observed_diff, p_value=p_value,
        ci_low=ci_low, ci_high=ci_high, n_boot=n_boot,
    )
