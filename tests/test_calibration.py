import pytest

from src.evaluation.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_diagram_bins,
)


def test_brier_score_perfect_predictions_is_zero():
    assert brier_score([1.0, 0.0, 1.0], [1.0, 0.0, 1.0]) == 0.0


def test_brier_score_worst_case_is_one():
    assert brier_score([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)


def test_brier_score_matches_hand_computed_value():
    # (0.9-1)^2 + (0.2-0)^2 = 0.01 + 0.04 = 0.05, /2 = 0.025
    assert brier_score([0.9, 0.2], [1.0, 0.0]) == pytest.approx(0.025)


def test_expected_calibration_error_is_zero_for_perfectly_calibrated_bins():
    # Every prediction in the [0.7, 0.8) bin has confidence ~0.75 and exactly
    # 75% of them are correct -> mean confidence == empirical accuracy.
    confidences = [0.75] * 4
    is_correct = [1.0, 1.0, 1.0, 0.0]
    ece = expected_calibration_error(confidences, is_correct, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_expected_calibration_error_penalizes_overconfidence():
    # Model says 0.95 confidence every time but is only right half the time -
    # badly overconfident, should produce a large ECE.
    confidences = [0.95] * 10
    is_correct = [1.0] * 5 + [0.0] * 5
    ece = expected_calibration_error(confidences, is_correct, n_bins=10)
    assert ece == pytest.approx(0.45, abs=1e-9)


def test_reliability_diagram_bins_counts_add_up():
    confidences = [0.05, 0.15, 0.55, 0.95, 0.98]
    is_correct = [0.0, 1.0, 1.0, 1.0, 0.0]
    bins = reliability_diagram_bins(confidences, is_correct, n_bins=10)
    assert sum(b.count for b in bins) == len(confidences)
