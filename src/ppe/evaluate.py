"""Detection metrics for the *shipped runtime path* (numpy only, no torch).

Ultralytics ``val`` scores a checkpoint with its own NMS. What a user gets is this package's decode
+ class-agnostic NMS on an ONNX model through some execution provider, so that is what gets scored
here. Every metric is reported with the counts behind it (boxes, TP/FP/FN) and a Wilson interval,
because a precision quoted from 43 boxes is not the same claim as one quoted from 4,300.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion (95% by default); None when n == 0."""
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def iou_one_to_many(box: Sequence[float], boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    b = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    return inter / (a + b - inter + 1e-9)


def read_yolo_labels(path: Path, width: int, height: int, names: Sequence[str]) -> list[tuple[str, tuple]]:
    """YOLO ``cls cx cy w h`` (normalized) → ``[(class_name, (x1, y1, x2, y2) in pixels)]``."""
    out: list[tuple[str, tuple]] = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        if not 0 <= cid < len(names):
            continue
        cx, cy, w, h = (float(v) for v in parts[1:5])
        out.append(
            (names[cid], ((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height))
        )
    return out


def label_path_for(image: Path) -> Path:
    """``.../images/x.jpg`` → ``.../labels/x.txt`` (the YOLO dataset convention)."""
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file())


@dataclass
class ClassScore:
    """Detections of one class over a dataset, matched to ground truth at a fixed IoU."""

    name: str
    n_gt: int
    conf: np.ndarray  # descending
    tp: np.ndarray  # 1 where the detection matched an unmatched GT box, same order as conf

    def ap50(self) -> float | None:
        """Area under the interpolated precision-recall curve, 101 points.

        Same convention as Ultralytics so the numbers are comparable; a consequence of it is that a
        perfect detector scores 0.995, not 1.0."""
        if self.n_gt == 0:
            return None
        if len(self.conf) == 0:
            return 0.0
        ctp = np.cumsum(self.tp)
        cfp = np.cumsum(1 - self.tp)
        recall = ctp / self.n_gt
        precision = ctp / (ctp + cfp)
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        x = np.linspace(0, 1, 101)
        y = np.interp(x, mrec, mpre)
        return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)))  # trapezoid rule

    def at(self, threshold: float, n_images: int) -> dict:
        """Counts and rates for detections with ``conf >= threshold``."""
        keep = self.conf >= threshold
        tp = int(self.tp[keep].sum())
        fp = int(keep.sum()) - tp
        fn = self.n_gt - tp
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / self.n_gt if self.n_gt else None
        p_ci = wilson_interval(tp, tp + fp)
        r_ci = wilson_interval(tp, self.n_gt)
        return {
            "conf": threshold,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": None if precision is None else round(precision, 4),
            "recall": None if recall is None else round(recall, 4),
            "precision_ci95": None if p_ci is None else [round(v, 4) for v in p_ci],
            "recall_ci95": None if r_ci is None else [round(v, 4) for v in r_ci],
            "fp_per_100_images": round(100.0 * fp / n_images, 2) if n_images else None,
        }


def score_dataset(
    predictions: Iterable[Sequence],
    ground_truth: Iterable[Sequence[tuple[str, tuple]]],
    class_names: Sequence[str],
    *,
    match_iou: float = 0.5,
) -> dict[str, ClassScore]:
    """Match per-image detections (objects with ``cls_name``, ``conf``, ``xyxy``) to labels.

    Standard greedy matching: within an image and class, detections are taken in descending
    confidence and each claims the best still-unmatched GT box with IoU >= ``match_iou``. Because the
    order is by confidence, the result restricted to ``conf >= t`` is exactly the matching at ``t``.
    """
    conf: dict[str, list[float]] = {n: [] for n in class_names}
    hits: dict[str, list[int]] = {n: [] for n in class_names}
    n_gt = dict.fromkeys(class_names, 0)
    for dets, labels in zip(predictions, ground_truth, strict=True):
        for name in class_names:
            gt = np.array([b for c, b in labels if c == name], dtype=np.float64).reshape(-1, 4)
            n_gt[name] += len(gt)
            taken = np.zeros(len(gt), dtype=bool)
            for det in sorted((d for d in dets if d.cls_name == name), key=lambda d: -d.conf):
                ious = iou_one_to_many(det.xyxy, gt)
                ious[taken] = -1.0
                j = int(np.argmax(ious)) if len(ious) else -1
                ok = j >= 0 and ious[j] >= match_iou
                if ok:
                    taken[j] = True
                conf[name].append(float(det.conf))
                hits[name].append(1 if ok else 0)
    out: dict[str, ClassScore] = {}
    for name in class_names:
        order = np.argsort(-np.asarray(conf[name], dtype=np.float64), kind="stable")
        out[name] = ClassScore(
            name=name,
            n_gt=n_gt[name],
            conf=np.asarray(conf[name], dtype=np.float64)[order],
            tp=np.asarray(hits[name], dtype=np.int64)[order],
        )
    return out


def evaluate_session(
    session,
    images: Sequence[Path],
    class_names: Sequence[str],
    *,
    match_iou: float = 0.5,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict:
    """Run ``session`` over labeled images and summarize. Unreadable images are counted, not hidden."""
    import cv2

    preds, gts, unreadable = [], [], 0
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            unreadable += 1
            continue
        h, w = image.shape[:2]
        preds.append(session.predict(image))
        gts.append(read_yolo_labels(label_path_for(Path(path)), w, h, class_names))
    summary = summarize_scores(score_dataset(preds, gts, class_names, match_iou=match_iou), len(preds), thresholds)
    summary["unreadable_images"] = unreadable
    return summary


def summarize_scores(
    scores: dict[str, ClassScore],
    n_images: int,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict:
    per_class = {}
    for name, s in scores.items():
        ap = s.ap50()
        per_class[name] = {
            "n_gt_boxes": s.n_gt,
            "n_detections": int(len(s.conf)),
            "ap50": None if ap is None else round(ap, 4),
            "sweep": [s.at(t, n_images) for t in thresholds],
        }
    aps = [v["ap50"] for v in per_class.values() if v["ap50"] is not None]
    return {"n_images": n_images, "map50": round(float(np.mean(aps)), 4) if aps else None, "per_class": per_class}
