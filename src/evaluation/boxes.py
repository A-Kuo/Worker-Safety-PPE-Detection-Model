"""IoU computation and greedy detection-to-ground-truth matching.

This is the foundation every other metric in this package builds on: mAP,
per-class P/R/F1, confusion matrices, and calibration all require first
deciding which predicted boxes are true positives vs. false positives, which
requires IoU-based matching against ground truth. Implemented from scratch
(no ultralytics/torchvision dependency) so it can run in any environment,
including this one (no GPU, no torch installed).
"""

from __future__ import annotations

from dataclasses import dataclass

# A box is (x1, y1, x2, y2) in any consistent coordinate system (pixels or
# normalized 0-1) - IoU is scale-invariant as long as both boxes use the same
# convention.
Box = tuple[float, float, float, float]


def iou(box_a: Box, box_b: Box) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


@dataclass
class Detection:
    box: Box
    score: float
    class_id: int


@dataclass
class GroundTruth:
    box: Box
    class_id: int


@dataclass
class MatchResult:
    """One row per prediction, in descending-score order, for one image."""

    detection: Detection
    is_true_positive: bool
    matched_gt_index: int | None  # index into the GroundTruth list, or None if FP
    iou: float  # IoU with the matched GT (0.0 if unmatched)


def match_detections(
    detections: list[Detection],
    ground_truths: list[GroundTruth],
    iou_threshold: float = 0.5,
) -> tuple[list[MatchResult], list[int]]:
    """Greedy TP/FP assignment for one image, standard COCO/YOLO-style protocol:

    - Only a detection and ground-truth box of the *same class* can match.
    - Process detections in descending confidence order.
    - Each ground-truth box can be matched to at most one detection (the
      highest-confidence detection that clears `iou_threshold` claims it).
    - A detection whose best available same-class IoU is below the threshold,
      or whose best match was already claimed by a higher-confidence
      detection, is a false positive.

    Returns (per-detection match results in the original input order,
    list of ground-truth indices that were never matched by anything = false
    negatives for this image).
    """
    order = sorted(range(len(detections)), key=lambda i: -detections[i].score)
    claimed_gt: set[int] = set()
    results: list[MatchResult | None] = [None] * len(detections)

    for i in order:
        det = detections[i]
        best_iou = 0.0
        best_gt_idx: int | None = None
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in claimed_gt or gt.class_id != det.class_id:
                continue
            this_iou = iou(det.box, gt.box)
            if this_iou > best_iou:
                best_iou = this_iou
                best_gt_idx = gt_idx

        if best_gt_idx is not None and best_iou >= iou_threshold:
            claimed_gt.add(best_gt_idx)
            results[i] = MatchResult(det, True, best_gt_idx, best_iou)
        else:
            results[i] = MatchResult(det, False, None, best_iou)

    unmatched_gt = [idx for idx in range(len(ground_truths)) if idx not in claimed_gt]
    return [r for r in results if r is not None], unmatched_gt
