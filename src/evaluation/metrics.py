"""Per-class precision/recall/F1 and confusion-matrix aggregation across a
whole dataset, built on top of the single-image matching in boxes.py.

Usage sketch (once we have a trained model to generate real predictions from):

    per_image_results = [match_detections(dets, gts, iou_threshold=0.5) for ...]
    agg = aggregate_matches(per_image_results, class_ids)
    for cls_id, pr in agg.per_class.items():
        print(class_names[cls_id], pr.precision, pr.recall, pr.f1)
"""

from __future__ import annotations

from dataclasses import dataclass

from src.evaluation.boxes import Detection, GroundTruth, MatchResult, match_detections


@dataclass
class PrecisionRecallF1:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class DatasetEvaluation:
    per_class: dict[int, PrecisionRecallF1]
    # confusion[true_class][predicted_class] = count, plus a synthetic
    # "background" bucket (-1) for false positives (predicted, no matching GT)
    # and false negatives (GT present, nothing predicted for it).
    confusion: dict[int, dict[int, int]]

    def micro_average(self) -> PrecisionRecallF1:
        tp = sum(c.tp for c in self.per_class.values())
        fp = sum(c.fp for c in self.per_class.values())
        fn = sum(c.fn for c in self.per_class.values())
        return PrecisionRecallF1(tp, fp, fn)

    def macro_f1(self) -> float:
        classes = list(self.per_class.values())
        return sum(c.f1 for c in classes) / len(classes) if classes else 0.0


BACKGROUND = -1


def aggregate_matches(
    per_image: list[tuple[list[MatchResult], list[int]]],
    ground_truths_per_image: list[list[GroundTruth]],
    class_ids: list[int],
) -> DatasetEvaluation:
    """`per_image[i]` is the (match_results, unmatched_gt_indices) tuple
    returned by `match_detections` for image i; `ground_truths_per_image[i]` is
    the same ground-truth list that was passed into that call (needed to look
    up the class of each false negative)."""
    counters = {c: {"tp": 0, "fp": 0, "fn": 0} for c in class_ids}
    confusion: dict[int, dict[int, int]] = {c: {} for c in class_ids}
    confusion[BACKGROUND] = {}

    for (match_results, unmatched_gt), gts in zip(per_image, ground_truths_per_image):
        for m in match_results:
            pred_class = m.detection.class_id
            if m.is_true_positive:
                counters[pred_class]["tp"] += 1
                confusion.setdefault(pred_class, {})
                confusion[pred_class][pred_class] = confusion[pred_class].get(pred_class, 0) + 1
            else:
                counters[pred_class]["fp"] += 1
                confusion[BACKGROUND][pred_class] = confusion[BACKGROUND].get(pred_class, 0) + 1

        for gt_idx in unmatched_gt:
            true_class = gts[gt_idx].class_id
            counters[true_class]["fn"] += 1
            confusion.setdefault(true_class, {})
            confusion[true_class][BACKGROUND] = confusion[true_class].get(BACKGROUND, 0) + 1

    per_class = {c: PrecisionRecallF1(v["tp"], v["fp"], v["fn"]) for c, v in counters.items()}
    return DatasetEvaluation(per_class=per_class, confusion=confusion)


def evaluate_dataset(
    detections_per_image: list[list[Detection]],
    ground_truths_per_image: list[list[GroundTruth]],
    class_ids: list[int],
    iou_threshold: float = 0.5,
) -> DatasetEvaluation:
    """Convenience wrapper: runs match_detections per image, then aggregates."""
    per_image = [
        match_detections(dets, gts, iou_threshold=iou_threshold)
        for dets, gts in zip(detections_per_image, ground_truths_per_image)
    ]
    return aggregate_matches(per_image, ground_truths_per_image, class_ids)
