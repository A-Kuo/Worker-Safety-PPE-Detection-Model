"""Calibration metrics for detector confidence scores: Expected Calibration
Error (ECE) and Brier score.

Both take the same two inputs: `confidences` (the model's predicted score for
each detection, after IoU matching) and `is_correct` (1.0 if that detection was
a true positive, 0.0 if it was a false positive) - i.e. run
`src/evaluation/boxes.match_detections` first, then feed
`[m.detection.score for m in results]` and `[1.0 if m.is_true_positive else 0.0
for m in results]` in here.

See docs/PROJECT_PLAN.md Section 4 (M3) for the formulas these implement.
"""

from __future__ import annotations

from dataclasses import dataclass


def brier_score(confidences: list[float], is_correct: list[float]) -> float:
    """BS = (1/N) * sum((p_i - y_i)^2). Lower is better; 0 = perfect
    calibration, 0.25 = the score of an uninformative p=0.5-always predictor
    against a 50/50 label distribution."""
    if len(confidences) != len(is_correct):
        raise ValueError("confidences and is_correct must be the same length")
    if not confidences:
        return 0.0
    n = len(confidences)
    return sum((p - y) ** 2 for p, y in zip(confidences, is_correct)) / n


@dataclass
class CalibrationBin:
    bin_lower: float
    bin_upper: float
    count: int
    mean_confidence: float
    empirical_accuracy: float


def reliability_diagram_bins(
    confidences: list[float],
    is_correct: list[float],
    n_bins: int = 10,
) -> list[CalibrationBin]:
    """Bucket predictions into `n_bins` equal-width confidence bins and compute
    the mean predicted confidence vs. empirical accuracy in each - the raw data
    behind both `expected_calibration_error` and a reliability-diagram plot."""
    bins: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for p, y in zip(confidences, is_correct):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, y))

    result = []
    for i, bucket in enumerate(bins):
        lower, upper = i / n_bins, (i + 1) / n_bins
        if not bucket:
            result.append(CalibrationBin(lower, upper, 0, 0.0, 0.0))
            continue
        mean_conf = sum(p for p, _ in bucket) / len(bucket)
        acc = sum(y for _, y in bucket) / len(bucket)
        result.append(CalibrationBin(lower, upper, len(bucket), mean_conf, acc))
    return result


def expected_calibration_error(
    confidences: list[float],
    is_correct: list[float],
    n_bins: int = 10,
) -> float:
    """ECE = sum_m (n_m/N) * |acc(m) - conf(m)|, per docs/PROJECT_PLAN.md."""
    bins = reliability_diagram_bins(confidences, is_correct, n_bins=n_bins)
    n = len(confidences)
    if n == 0:
        return 0.0
    return sum(
        (b.count / n) * abs(b.empirical_accuracy - b.mean_confidence) for b in bins
    )
