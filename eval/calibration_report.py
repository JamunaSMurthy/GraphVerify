"""
Per-dataset, per-class calibration reporting: ECE tables plus reliability
diagrams, so calibration is inspectable evidence (revision plan §4.4/§9)
rather than a single aggregate number that could hide class-specific
miscalibration.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from .metrics import per_class_ece


def build_calibration_table(
    dataset_predictions: Dict[str, List[Dict[str, Any]]],
    n_bins: int = 15,
) -> Dict[str, Dict[str, float]]:
    """
    `dataset_predictions` maps dataset name -> list of claim-level
    prediction dicts, each needing ``"verdict"``, ``"gold_verdict"``, and
    ``"reliability"``. Returns dataset -> {"overall", "Supported",
    "Unsupported", "Contradictory"} ECE values (a class entry is omitted
    for a dataset with no predicted claims of that class).
    """
    table = {}
    for dataset, records in dataset_predictions.items():
        preds  = [r["verdict"] for r in records]
        golds  = [r["gold_verdict"] for r in records]
        scores = [float(r["reliability"]) for r in records]
        table[dataset] = per_class_ece(preds, golds, scores, n_bins=n_bins)
    return table


def reliability_diagram(
    preds: List[str],
    golds: List[str],
    scores: List[float],
    n_bins: int = 15,
    title: str = "Reliability diagram",
    output_path: Optional[str] = None,
):
    """
    Plots a standard reliability diagram (binned mean confidence vs. binned
    accuracy, with a bin-count histogram beneath it) using matplotlib.
    Saves to `output_path` (format inferred from extension, e.g. .png/.svg)
    if given and returns that path; otherwise returns the matplotlib
    Figure for the caller to save/show/embed. Bins with zero samples are
    omitted from the plotted calibration line (there is nothing to plot)
    but still appear as empty bars in the count histogram, so a reviewer
    can see which parts of the curve are estimated from few points.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assert len(preds) == len(golds) == len(scores)
    correct = np.array([1 if p == g else 0 for p, g in zip(preds, golds)], dtype=np.float64)
    scores_arr = np.clip(np.array(scores, dtype=np.float64), 0.0, 1.0)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_acc: List[float] = []
    bin_conf: List[float] = []
    bin_count: List[int] = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (scores_arr >= lo) & (scores_arr < hi)
        if mask.sum() == 0:
            bin_acc.append(float("nan"))
            bin_conf.append(float((lo + hi) / 2))
            bin_count.append(0)
            continue
        bin_acc.append(float(correct[mask].mean()))
        bin_conf.append(float(scores_arr[mask].mean()))
        bin_count.append(int(mask.sum()))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(5, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]},
    )
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    valid = [i for i, a in enumerate(bin_acc) if not np.isnan(a)]
    ax1.plot([bin_conf[i] for i in valid], [bin_acc[i] for i in valid], marker="o", label="Model")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1)
    ax1.set_title(title)
    ax1.legend()

    ax2.bar(bin_edges[:-1], bin_count, width=1.0 / n_bins, align="edge", color="steelblue")
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Count")

    fig.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path
    return fig
