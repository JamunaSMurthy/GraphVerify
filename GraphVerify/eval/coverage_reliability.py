"""
Answer-level coverage/abstention analysis.

The revision plan flags a specific risk with answer-level results: a
verifier can look like it "improves downstream reliability" purely by
rejecting the hardest answers, inflating EM/F1 on the accepted subset
through selection bias rather than genuine faithfulness improvement. The
fix is to always report coverage alongside reliability — the fraction of
answers accepted/rejected at each confidence threshold, EM/F1 on the
accepted subset, and EM/F1 on the full set — so a reviewer can see whether
gains survive abstention accounting.

Acceptance policy (per the revision plan's draft coverage paragraph, §13.3):
an answer is accepted only when every one of its claims is either (a)
Supported with reliability at or above the threshold, or (b) Unsupported
(missing evidence is treated as "explicitly marked unverifiable and
excluded", not as grounds to reject the whole answer). Any Contradictory
claim, or any Supported claim whose reliability falls below the threshold,
causes the whole answer to be rejected. This makes acceptance rate strictly
non-increasing as the threshold rises, the standard shape of a
risk-coverage curve.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def is_claim_acceptable(verdict: str, reliability: float, threshold: float) -> bool:
    """Per-claim acceptability check used by :func:`accept_answer` — see module docstring for the policy."""
    if verdict == "Contradictory":
        return False
    if verdict == "Unsupported":
        return True
    if verdict == "Supported":
        return reliability >= threshold
    return False


def accept_answer(claim_records: Sequence[Dict[str, Any]], threshold: float) -> bool:
    """
    An answer is accepted iff every one of its claims is acceptable at
    `threshold` (see :func:`is_claim_acceptable`). An answer with zero
    claims (e.g. empty generation, or decomposition returned nothing) is
    vacuously accepted — callers computing coverage over a dataset should
    decide whether to exclude such answers before calling this, since a
    zero-claim "acceptance" carries no reliability signal.
    """
    return all(
        is_claim_acceptable(r.get("verdict", ""), float(r.get("reliability", 0.0)), threshold)
        for r in claim_records
    )


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")


def normalize_answer(text: str) -> str:
    """Standard SQuAD-style answer normalization: lowercase, strip punctuation/articles, collapse whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def f1_score(prediction: str, ground_truth: str) -> float:
    """Standard SQuAD-style token-overlap F1 between a predicted and gold answer string."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


@dataclass
class CoverageResult:
    threshold:     float
    accepted_pct:  float
    rejected_pct:  float
    accepted_em:   float
    accepted_f1:   float
    full_set_em:   float
    full_set_f1:   float
    n_answers:     int
    n_accepted:    int


def coverage_reliability_report(
    answers: List[Dict[str, Any]],
    thresholds: Sequence[float] = (0.50, 0.60, 0.70, 0.80, 0.90),
) -> List[CoverageResult]:
    """
    Computes the coverage/abstention table (Table 2/3/14-style) across a
    sweep of reliability thresholds.

    Each element of `answers` needs: ``"generated"`` (the generated answer
    text), ``"gold_answer"`` (reference answer for EM/F1), and
    ``"claim_records"`` (that answer's list of per-claim verdict/reliability
    dicts, as produced by :func:`graphverify.verdict_assigner.record_to_dict`
    or a baseline's equivalent).
    """
    full_em = [exact_match_score(a["generated"], a["gold_answer"]) for a in answers]
    full_f1 = [f1_score(a["generated"], a["gold_answer"]) for a in answers]
    n = len(answers)

    results = []
    for threshold in thresholds:
        accepted_mask = [accept_answer(a["claim_records"], threshold) for a in answers]
        n_accepted = sum(accepted_mask)
        acc_em = [em for em, keep in zip(full_em, accepted_mask) if keep]
        acc_f1 = [f1 for f1, keep in zip(full_f1, accepted_mask) if keep]

        results.append(CoverageResult(
            threshold=threshold,
            accepted_pct=100.0 * n_accepted / n if n else 0.0,
            rejected_pct=100.0 * (n - n_accepted) / n if n else 0.0,
            accepted_em=100.0 * float(np.mean(acc_em)) if acc_em else 0.0,
            accepted_f1=100.0 * float(np.mean(acc_f1)) if acc_f1 else 0.0,
            full_set_em=100.0 * float(np.mean(full_em)) if full_em else 0.0,
            full_set_f1=100.0 * float(np.mean(full_f1)) if full_f1 else 0.0,
            n_answers=n, n_accepted=n_accepted,
        ))
    return results


def hallucination_precision_recall(
    scores: List[float],
    is_hallucination: Sequence[int],
    threshold: float,
) -> Dict[str, float]:
    """
    Precision/recall/F1 for hallucination detection at a fixed decision
    threshold: a response is flagged as hallucinated when
    ``1 - reliability >= threshold`` (i.e. reliability drops low enough).
    Complements :func:`eval.metrics.hallucination_auroc_auprc`, which is
    threshold-free.
    """
    assert len(scores) == len(is_hallucination)
    risk_scores = 1.0 - np.asarray(scores, dtype=np.float64)
    flagged = (risk_scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        is_hallucination, flagged, average="binary", zero_division=0,
    )
    return {"precision": float(precision) * 100, "recall": float(recall) * 100, "f1": float(f1) * 100}
