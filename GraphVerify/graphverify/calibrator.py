"""
Temperature scaling calibrator for reliability scores.

Learns a single scalar T on the validation split that minimises
Expected Calibration Error (ECE).

Calibrated score = sigmoid(logit(raw_score) / T)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class CalibrationResult:
    temperature: float = 1.0
    ece_before:  float = 0.0
    ece_after:   float = 0.0
    n_samples:   int   = 0


class TemperatureCalibrator:

    def __init__(self, n_bins: int = 15) -> None:
        self.n_bins = n_bins
        self.temperature: float = 1.0

    def fit(self, scores: List[float], labels: List[int]) -> CalibrationResult:
        """Learn temperature on validation data. labels: 1 = correct, 0 = incorrect."""
        scores_arr = np.clip(np.array(scores, dtype=np.float64), 1e-6, 1 - 1e-6)
        labels_arr = np.array(labels, dtype=np.float64)

        ece_before = self._ece(scores_arr, labels_arr)

        def nll(T):
            cal = self._calibrate(scores_arr, T)
            cal = np.clip(cal, 1e-9, 1 - 1e-9)
            return -np.mean(
                labels_arr * np.log(cal) + (1 - labels_arr) * np.log(1 - cal)
            )

        res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.temperature = float(res.x)

        cal_scores = self._calibrate(scores_arr, self.temperature)
        return CalibrationResult(
            temperature=self.temperature,
            ece_before=ece_before,
            ece_after=self._ece(cal_scores, labels_arr),
            n_samples=len(scores),
        )

    def calibrate(self, score: float) -> float:
        s = np.clip(score, 1e-6, 1 - 1e-6)
        return float(self._calibrate(np.array([s]), self.temperature)[0])

    def _calibrate(self, scores: np.ndarray, T: float) -> np.ndarray:
        logits = np.log(scores / (1.0 - scores))
        return 1.0 / (1.0 + np.exp(-logits / T))

    def _ece(self, scores: np.ndarray, labels: np.ndarray) -> float:
        n = len(scores)
        bins = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (scores >= lo) & (scores < hi)
            if mask.sum() == 0:
                continue
            ece += mask.sum() / n * abs(scores[mask].mean() - labels[mask].mean())
        return float(ece)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"temperature": self.temperature, "n_bins": self.n_bins}, f)

    def load(self, path: str) -> None:
        with open(path) as f:
            d = json.load(f)
        self.temperature = float(d.get("temperature", 1.0))
        self.n_bins      = int(d.get("n_bins", 15))


def compute_ece(scores: List[float], labels: List[int], n_bins: int = 15) -> float:
    cal = TemperatureCalibrator(n_bins=n_bins)
    return cal._ece(
        np.clip(np.array(scores, dtype=np.float64), 1e-6, 1 - 1e-6),
        np.array(labels, dtype=np.float64),
    )
