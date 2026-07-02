"""Tests for graphverify/calibrator.py: ECE computation and temperature scaling."""
from __future__ import annotations

import pytest

from graphverify.calibrator import TemperatureCalibrator, compute_ece


def test_perfect_calibration_has_zero_ece():
    # scores exactly equal to the empirical accuracy in each bin -> ECE ~ 0
    # (compute_ece clips scores into [1e-6, 1-1e-6] to avoid log(0) in
    # temperature fitting elsewhere, so exact 0.0 is not attainable).
    scores = [0.0, 0.0, 1.0, 1.0]
    labels = [0, 0, 1, 1]
    ece = compute_ece(scores, labels, n_bins=2)
    assert ece == pytest.approx(0.0, abs=1e-4)


def test_maximally_overconfident_wrong_predictions_have_high_ece():
    scores = [1.0, 1.0, 1.0, 1.0]
    labels = [0, 0, 0, 0]  # always confident, always wrong
    ece = compute_ece(scores, labels, n_bins=10)
    assert ece == pytest.approx(1.0, abs=1e-4)


def test_ece_is_bounded_in_zero_one():
    scores = [0.1, 0.4, 0.6, 0.9, 0.99]
    labels = [1, 0, 1, 0, 1]
    ece = compute_ece(scores, labels)
    assert 0.0 <= ece <= 1.0


def test_temperature_calibrator_fit_returns_valid_result():
    # Temperature scaling minimizes NLL, not ECE directly, so ECE is not
    # guaranteed to strictly decrease on a small/noisy sample -- this test
    # checks the fit produces a valid, sane result rather than asserting a
    # monotonic ECE improvement that the underlying method does not promise.
    calibrator = TemperatureCalibrator(n_bins=10)
    scores = [0.95, 0.9, 0.92, 0.1, 0.05, 0.5, 0.6]
    labels = [1, 0, 1, 0, 0, 1, 0]  # deliberately miscalibrated (overconfident)
    result = calibrator.fit(scores, labels)
    assert result.temperature > 0
    assert 0.0 <= result.ece_before <= 1.0
    assert 0.0 <= result.ece_after <= 1.0
    assert result.n_samples == len(scores)


def test_temperature_calibrator_fit_reduces_ece_on_clear_overconfidence():
    # A larger, unambiguously overconfident sample (scores near 1.0 but only
    # half correct) is where NLL-minimizing temperature scaling reliably
    # also reduces ECE.
    calibrator = TemperatureCalibrator(n_bins=10)
    scores = [0.95] * 20
    labels = [1, 0] * 10  # 50% accuracy despite 95% confidence
    result = calibrator.fit(scores, labels)
    assert result.ece_after < result.ece_before


def test_calibrate_single_score_uses_fitted_temperature():
    calibrator = TemperatureCalibrator()
    calibrator.temperature = 2.0
    calibrated = calibrator.calibrate(0.9)
    assert 0.0 <= calibrated <= 1.0
    # temperature > 1 should pull an extreme score toward 0.5
    assert calibrated < 0.9


def test_save_and_load_roundtrip(tmp_path):
    calibrator = TemperatureCalibrator(n_bins=20)
    calibrator.temperature = 1.5
    path = str(tmp_path / "calibrator.json")
    calibrator.save(path)

    loaded = TemperatureCalibrator()
    loaded.load(path)
    assert loaded.temperature == pytest.approx(1.5)
    assert loaded.n_bins == 20
