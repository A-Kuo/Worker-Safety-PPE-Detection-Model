"""Numpy-only pre/post-processing for YOLO-style detectors.

An exported ONNX graph gives raw tensors, not boxes. These helpers do the
resize, decode, and NMS that Ultralytics normally hides, which keeps the ONNX
runtime path free of a torch dependency and makes each step testable on its
own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LetterboxInfo:
    """Scale and padding applied when fitting an image into a square canvas."""

    scale: float
    pad_x: float
    pad_y: float
    src_width: int
    src_height: int


def letterbox(
    image: np.ndarray,
    size: int = 640,
    fill: int = 114,
) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize with a preserved aspect ratio and pad to ``size`` x ``size``."""
    if image.ndim != 3:
        raise ValueError(f"Expected an HxWxC image, got shape {image.shape}")
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("Cannot letterbox an empty image")

    scale = min(size / height, size / width)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    pad_x = (size - new_w) / 2.0
    pad_y = (size - new_h) / 2.0

    resized = _resize(image, new_w, new_h)
    canvas = np.full((size, size, image.shape[2]), fill, dtype=image.dtype)
    top = int(round(pad_y - 0.1))
    left = int(round(pad_x - 0.1))
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, LetterboxInfo(scale, float(left), float(top), width, height)


def to_input_tensor(canvas: np.ndarray) -> np.ndarray:
    """Turn an HWC BGR uint8 canvas into a 1x3xHxW float32 RGB batch."""
    rgb = canvas[:, :, ::-1]
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return np.ascontiguousarray(chw[None, ...])


def undo_letterbox(boxes: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    """Map xyxy boxes from canvas coordinates back onto the source image."""
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    out = boxes.astype(np.float32).copy()
    out[:, [0, 2]] -= info.pad_x
    out[:, [1, 3]] -= info.pad_y
    out /= info.scale
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, info.src_width)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, info.src_height)
    return out


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert center-form boxes to corner form."""
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression over xyxy boxes, highest score first."""
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        best = int(order[0])
        keep.append(best)
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[best], x1[rest])
        iy1 = np.maximum(y1[best], y1[rest])
        ix2 = np.minimum(x2[best], x2[rest])
        iy2 = np.minimum(y2[best], y2[rest])
        inter = (ix2 - ix1).clip(min=0) * (iy2 - iy1).clip(min=0)
        union = areas[best] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= iou_threshold]
    return keep


def decode_yolo_output(
    raw: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int = 300,
    num_classes: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a YOLOv8 head into ``(boxes_xyxy, scores, class_ids)``.

    The exported tensor is ``(1, 4 + num_classes, num_anchors)``: four box
    values per anchor followed by one score per class. Some exports transpose
    the last two axes, so pass ``num_classes`` when you know it and the layout
    is resolved exactly instead of by guessing which axis is longer.
    """
    array = np.asarray(raw, dtype=np.float32)
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D or 3D detection tensor, got shape {raw.shape}")
    array = _orient(array, num_classes)
    if array.shape[0] < 5:
        raise ValueError(f"Detection tensor needs at least 5 rows, got {array.shape[0]}")

    boxes = xywh_to_xyxy(array[:4].T)
    class_scores = array[4:]
    class_ids = class_scores.argmax(axis=0)
    scores = class_scores.max(axis=0)

    above = scores >= conf_threshold
    boxes, scores, class_ids = boxes[above], scores[above], class_ids[above]
    if boxes.shape[0] == 0:
        return boxes.reshape(0, 4), scores, class_ids

    keep = _class_aware_nms(boxes, scores, class_ids, iou_threshold)[:max_detections]
    return boxes[keep], scores[keep], class_ids[keep]


def _orient(array: np.ndarray, num_classes: int | None) -> np.ndarray:
    """Put channels on axis 0, anchors on axis 1."""
    if num_classes is not None:
        channels = 4 + int(num_classes)
        if array.shape[0] != channels and array.shape[1] == channels:
            return array.T
        return array
    return array.T if array.shape[0] > array.shape[1] else array


def _class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    """Suppress within each class by offsetting boxes into disjoint strips."""
    span = float(boxes.max() - boxes.min()) + 1.0 if boxes.size else 1.0
    offsets = class_ids.astype(np.float32)[:, None] * span
    keep = nms(boxes + offsets, scores, iou_threshold)
    return sorted(keep, key=lambda i: -scores[i])


def _resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        return _resize_nearest(image, width, height)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def _resize_nearest(image: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    rows = (np.arange(height) * (src_h / height)).astype(np.int64).clip(0, src_h - 1)
    cols = (np.arange(width) * (src_w / width)).astype(np.int64).clip(0, src_w - 1)
    return image[rows][:, cols]
