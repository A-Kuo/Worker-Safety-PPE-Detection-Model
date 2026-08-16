"""Confidence-threshold selection for safety-critical classes.

The framing (docs/PROJECT_PLAN.md M3): for a class like `no_helmet`, missing a
real violation (false negative) is worse than a false alarm (false positive).
So instead of picking the threshold that maximizes F1 (which weighs precision
and recall equally), we pick the *lowest* threshold that still satisfies a
minimum recall constraint, and report the precision/false-positive cost that
comes with it - making the trade-off explicit rather than hiding it inside a
single F1 number.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import precision_recall_curve


@dataclass
class OperatingPoint:
    threshold: float
    precision: float
    recall: float

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def best_f1_operating_point(y_true: list[int], scores: list[float]) -> OperatingPoint:
    """The "default" choice: maximize F1. Useful as a baseline to compare the
    safety-constrained choice below against."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    best = OperatingPoint(threshold=0.0, precision=0.0, recall=0.0)
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        candidate = OperatingPoint(threshold=float(t), precision=float(p), recall=float(r))
        if candidate.f1 > best.f1:
            best = candidate
    return best


def min_recall_operating_point(
    y_true: list[int],
    scores: list[float],
    min_recall: float,
) -> OperatingPoint:
    """The *highest* confidence threshold that still achieves recall >=
    `min_recall` - i.e. the most precise (fewest false alarms) operating point
    available without violating the safety floor.

    Raises ValueError if no threshold in the data achieves `min_recall` (can
    happen if the model's overall recall ceiling is below the target - that's
    a real, important finding to surface, not something to silently clamp).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision_recall_curve returns points in order of decreasing threshold
    # (i.e. increasing recall); thresholds has one fewer element than
    # precision/recall because the last point is always (recall=0 or 1,
    # precision=..., no corresponding threshold).
    candidates = [
        OperatingPoint(threshold=float(t), precision=float(p), recall=float(r))
        for p, r, t in zip(precision[:-1], recall[:-1], thresholds)
        if r >= min_recall
    ]
    if not candidates:
        raise ValueError(
            f"No threshold in the provided scores achieves recall >= {min_recall}. "
            f"Max achievable recall in this data: {max(recall):.4f}. This means the "
            f"model itself, not just the threshold choice, needs improvement to hit "
            f"this safety target."
        )
    # Among all thresholds that satisfy the recall floor, prefer the one with
    # the highest threshold (== highest precision among them, since precision
    # is monotonically non-decreasing as threshold increases for a
    # well-behaved classifier) - i.e. the strictest threshold that still clears
    # the bar.
    return max(candidates, key=lambda c: c.threshold)
