import pytest

from src.evaluation.threshold_optimization import (
    best_f1_operating_point,
    min_recall_operating_point,
)


def _synthetic_scores():
    # 10 positives, 10 negatives, scores separate them imperfectly so there's
    # a real precision/recall trade-off to navigate (not a trivial 0/1 split).
    y_true = [1] * 10 + [0] * 10
    scores = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.55, 0.4, 0.3] + [
        0.65, 0.5, 0.45, 0.35, 0.25, 0.2, 0.15, 0.1, 0.05, 0.02
    ]
    return y_true, scores


def test_best_f1_operating_point_beats_extreme_thresholds():
    y_true, scores = _synthetic_scores()
    best = best_f1_operating_point(y_true, scores)
    assert 0.0 < best.precision <= 1.0
    assert 0.0 < best.recall <= 1.0
    assert best.f1 > 0.5  # scores are reasonably well-separated


def test_min_recall_operating_point_satisfies_the_floor():
    y_true, scores = _synthetic_scores()
    op = min_recall_operating_point(y_true, scores, min_recall=0.9)
    assert op.recall >= 0.9


def test_min_recall_operating_point_is_stricter_than_best_f1_when_recall_target_is_high():
    y_true, scores = _synthetic_scores()
    safety_point = min_recall_operating_point(y_true, scores, min_recall=0.9)
    f1_point = best_f1_operating_point(y_true, scores)
    # Forcing recall=0.9 (higher than whatever recall the F1-optimal point
    # happens to land on) should require an equal-or-lower confidence
    # threshold, i.e. it must be at least as permissive.
    assert safety_point.recall >= f1_point.recall


def test_min_recall_operating_point_raises_when_unachievable():
    y_true, scores = _synthetic_scores()
    with pytest.raises(ValueError):
        min_recall_operating_point(y_true, scores, min_recall=1.01)
