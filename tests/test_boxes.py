import pytest

from src.evaluation.boxes import Detection, GroundTruth, iou, match_detections


def test_iou_identical_boxes_is_one():
    box = (0.0, 0.0, 10.0, 10.0)
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    a = (0.0, 0.0, 5.0, 5.0)
    b = (10.0, 10.0, 15.0, 15.0)
    assert iou(a, b) == 0.0


def test_iou_half_overlap():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 0.0, 15.0, 10.0)
    # intersection = 5x10=50, union = 100+100-50=150
    assert iou(a, b) == pytest.approx(50 / 150)


def test_match_detections_simple_true_positive():
    det = Detection(box=(0, 0, 10, 10), score=0.9, class_id=0)
    gt = GroundTruth(box=(0, 0, 10, 10), class_id=0)
    results, unmatched_gt = match_detections([det], [gt], iou_threshold=0.5)
    assert len(results) == 1
    assert results[0].is_true_positive
    assert unmatched_gt == []


def test_match_detections_wrong_class_is_false_positive():
    det = Detection(box=(0, 0, 10, 10), score=0.9, class_id=1)
    gt = GroundTruth(box=(0, 0, 10, 10), class_id=0)
    results, unmatched_gt = match_detections([det], [gt], iou_threshold=0.5)
    assert results[0].is_true_positive is False
    assert unmatched_gt == [0]  # the GT box was never matched -> false negative


def test_match_detections_low_iou_is_false_positive():
    det = Detection(box=(0, 0, 2, 2), score=0.9, class_id=0)
    gt = GroundTruth(box=(0, 0, 10, 10), class_id=0)
    results, unmatched_gt = match_detections([det], [gt], iou_threshold=0.5)
    assert results[0].is_true_positive is False
    assert unmatched_gt == [0]


def test_match_detections_higher_confidence_claims_the_gt_box():
    # Two overlapping detections of the same class competing for one GT box -
    # the higher-confidence one should win the match; the loser becomes a FP
    # even though it also overlaps the same GT box.
    gt = GroundTruth(box=(0, 0, 10, 10), class_id=0)
    strong = Detection(box=(0, 0, 10, 10), score=0.95, class_id=0)
    weak = Detection(box=(0, 0, 10, 10), score=0.4, class_id=0)
    results, unmatched_gt = match_detections([weak, strong], [gt], iou_threshold=0.5)

    by_score = {r.detection.score: r for r in results}
    assert by_score[0.95].is_true_positive
    assert by_score[0.4].is_true_positive is False
    assert unmatched_gt == []


def test_match_detections_no_predictions_all_gt_are_false_negatives():
    gts = [GroundTruth(box=(0, 0, 10, 10), class_id=0), GroundTruth(box=(20, 20, 30, 30), class_id=1)]
    results, unmatched_gt = match_detections([], gts, iou_threshold=0.5)
    assert results == []
    assert unmatched_gt == [0, 1]
