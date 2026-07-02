"""Tests for eval/calibration_report.py."""
from __future__ import annotations

import os

from eval.calibration_report import build_calibration_table, reliability_diagram


def test_build_calibration_table_per_dataset():
    dataset_predictions = {
        "hotpotqa": [
            {"verdict": "Supported", "gold_verdict": "Supported", "reliability": 0.9},
            {"verdict": "Unsupported", "gold_verdict": "Supported", "reliability": 0.4},
        ],
        "fever": [
            {"verdict": "Contradictory", "gold_verdict": "Contradictory", "reliability": 0.8},
        ],
    }
    table = build_calibration_table(dataset_predictions)
    assert "hotpotqa" in table and "fever" in table
    assert "overall" in table["hotpotqa"]


def test_reliability_diagram_saves_to_file(tmp_path):
    preds = ["Supported"] * 10 + ["Unsupported"] * 10
    golds = ["Supported"] * 15 + ["Unsupported"] * 5
    scores = [0.9] * 10 + [0.2] * 10
    out_path = str(tmp_path / "reliability.png")
    result = reliability_diagram(preds, golds, scores, output_path=out_path)
    assert result == out_path
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_reliability_diagram_returns_figure_without_output_path():
    preds = ["Supported"] * 5
    golds = ["Supported"] * 5
    scores = [0.9] * 5
    fig = reliability_diagram(preds, golds, scores)
    assert fig is not None
