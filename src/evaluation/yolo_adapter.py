"""Glue between Ultralytics YOLO's I/O formats and the from-scratch metrics in
boxes.py / metrics.py / calibration.py / threshold_optimization.py.

Split deliberately so the label-parsing/coordinate-conversion logic (pure
Python, no dependencies beyond Pillow) is unit-testable in this environment
(no GPU, no torch/ultralytics installed here), while the one function that
actually needs a trained model (`run_yolo_predictions`) lazily imports
`ultralytics` so importing this module doesn't fail where it isn't installed.

Intended usage, once you have a trained best.pt from a Kaggle/Colab run (see
notebooks/train_on_kaggle_or_colab.ipynb):

    from src.evaluation.yolo_adapter import evaluate_yolo_val_split

    result, raw = evaluate_yolo_val_split(
        model_path="runs/detect/train/weights/best.pt",
        images_dir="data/unified/val/images",
        labels_dir="data/unified/val/labels",
        class_ids=list(range(len(UNIFIED_CLASSES))),
    )
    for cls_id, pr in result.per_class.items():
        print(UNIFIED_CLASSES[cls_id], pr.precision, pr.recall, pr.f1)

    # raw["confidences"] / raw["is_correct"] feed directly into
    # src.evaluation.calibration.expected_calibration_error /
    # src.evaluation.threshold_optimization.min_recall_operating_point.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.evaluation.boxes import Box, Detection, GroundTruth, match_detections
from src.evaluation.metrics import DatasetEvaluation, aggregate_matches

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def yolo_label_to_ground_truths(label_path: Path, img_width: int, img_height: int) -> list[GroundTruth]:
    """Parse one YOLO-format label file (normalized `class cx cy w h` per
    line) into pixel-space GroundTruth boxes."""
    if not label_path.exists():
        return []
    ground_truths = []
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            class_id = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:5])
            box: Box = (
                (cx - w / 2) * img_width,
                (cy - h / 2) * img_height,
                (cx + w / 2) * img_width,
                (cy + h / 2) * img_height,
            )
            ground_truths.append(GroundTruth(box=box, class_id=class_id))
    return ground_truths


def find_image_for_stem(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = images_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def run_yolo_predictions(
    model_path: str,
    image_paths: list[Path],
    conf: float = 0.001,
) -> dict[str, list[Detection]]:
    """Run a trained YOLO model over a list of images and return per-image
    Detection lists, keyed by image stem. `conf=0.001` (near-zero) is
    intentional: we want *all* candidate detections so threshold selection can
    happen afterwards in threshold_optimization.py, not have Ultralytics'
    default 0.25 confidence cutoff silently throw away the low-confidence
    detections a safety-recall analysis needs to see.

    Requires `ultralytics` + `torch` to be installed - not available in this
    cloud agent environment, hence the lazy import. Run this from
    notebooks/train_on_kaggle_or_colab.ipynb instead.
    """
    from ultralytics import YOLO  # noqa: PLC0415 - intentionally lazy, see docstring

    model = YOLO(model_path)
    predictions: dict[str, list[Detection]] = {}
    for image_path in image_paths:
        result = model.predict(source=str(image_path), conf=conf, verbose=False)[0]
        dets = [
            Detection(box=tuple(box.tolist()), score=float(score), class_id=int(cls))
            for box, score, cls in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
            )
        ]
        predictions[image_path.stem] = dets
    return predictions


def evaluate_yolo_val_split(
    model_path: str,
    images_dir: str | Path,
    labels_dir: str | Path,
    class_ids: list[int],
    iou_threshold: float = 0.5,
    conf: float = 0.001,
) -> tuple[DatasetEvaluation, dict[str, list[float]]]:
    """End-to-end: run predictions, match against ground truth, aggregate.

    Returns (DatasetEvaluation, raw) where `raw` has:
      - "confidences" / "is_correct": flattened across every image/detection
        and every class - feed directly into calibration.py.
      - "per_class": {class_id: {"confidences": [...], "is_correct": [...]}} -
        feed a single class's lists into threshold_optimization.py for a
        safety-critical-class analysis (is_correct doubles as y_true: 1.0 for
        a true positive of that class, 0.0 for a false positive of that class).
    """
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    label_paths = sorted(labels_dir.glob("*.txt"))

    image_paths: list[Path] = []
    ground_truths_per_image: list[list[GroundTruth]] = []
    for label_path in label_paths:
        image_path = find_image_for_stem(images_dir, label_path.stem)
        if image_path is None:
            continue
        with Image.open(image_path) as img:
            w, h = img.size
        image_paths.append(image_path)
        ground_truths_per_image.append(yolo_label_to_ground_truths(label_path, w, h))

    predictions_by_stem = run_yolo_predictions(model_path, image_paths, conf=conf)
    detections_per_image = [predictions_by_stem[p.stem] for p in image_paths]

    per_image_matches = [
        match_detections(dets, gts, iou_threshold=iou_threshold)
        for dets, gts in zip(detections_per_image, ground_truths_per_image)
    ]
    result = aggregate_matches(per_image_matches, ground_truths_per_image, class_ids)

    confidences: list[float] = []
    is_correct: list[float] = []
    per_class: dict[int, dict[str, list[float]]] = {c: {"confidences": [], "is_correct": []} for c in class_ids}
    for match_results, _ in per_image_matches:
        for m in match_results:
            confidences.append(m.detection.score)
            correct = 1.0 if m.is_true_positive else 0.0
            is_correct.append(correct)
            per_class[m.detection.class_id]["confidences"].append(m.detection.score)
            per_class[m.detection.class_id]["is_correct"].append(correct)

    return result, {"confidences": confidences, "is_correct": is_correct, "per_class": per_class}
