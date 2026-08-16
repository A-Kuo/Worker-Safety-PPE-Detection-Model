import pytest

from src.evaluation.boxes import Detection, GroundTruth
from src.evaluation.metrics import PrecisionRecallF1, evaluate_dataset


def test_precision_recall_f1_properties():
    pr = PrecisionRecallF1(tp=8, fp=2, fn=2)
    assert pr.precision == pytest.approx(0.8)
    assert pr.recall == pytest.approx(0.8)
    assert pr.f1 == pytest.approx(0.8)


def test_precision_recall_f1_handles_zero_denominators():
    pr = PrecisionRecallF1(tp=0, fp=0, fn=0)
    assert pr.precision == 0.0
    assert pr.recall == 0.0
    assert pr.f1 == 0.0


def test_evaluate_dataset_two_images_mixed_outcomes():
    # Image 1: perfect detection of class 0.
    # Image 2: a false-positive class-1 detection, and a missed class-0 GT box
    # (false negative).
    detections = [
        [Detection(box=(0, 0, 10, 10), score=0.9, class_id=0)],
        [Detection(box=(50, 50, 60, 60), score=0.6, class_id=1)],
    ]
    ground_truths = [
        [GroundTruth(box=(0, 0, 10, 10), class_id=0)],
        [GroundTruth(box=(0, 0, 10, 10), class_id=0)],
    ]

    result = evaluate_dataset(detections, ground_truths, class_ids=[0, 1], iou_threshold=0.5)

    assert result.per_class[0].tp == 1
    assert result.per_class[0].fn == 1
    assert result.per_class[0].fp == 0

    assert result.per_class[1].fp == 1
    assert result.per_class[1].tp == 0

    micro = result.micro_average()
    assert micro.tp == 1
    assert micro.fp == 1
    assert micro.fn == 1
