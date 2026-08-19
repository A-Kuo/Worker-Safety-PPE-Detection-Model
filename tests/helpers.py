"""Small builders shared between test modules."""

from __future__ import annotations

import numpy as np

from ppe.backends import RawDetection
from ppe.schema import UNIFIED_CLASS_NAMES

PERSON_BOX = (100.0, 60.0, 220.0, 400.0)
HELMET_BOX = (130.0, 62.0, 190.0, 110.0)
VEST_BOX = (110.0, 150.0, 210.0, 280.0)
SECOND_PERSON_BOX = (320.0, 70.0, 440.0, 410.0)


def detection(name: str, box: tuple[float, float, float, float], conf: float = 0.9) -> RawDetection:
    """A RawDetection carrying a unified class name."""
    return RawDetection(
        cls_id=UNIFIED_CLASS_NAMES.index(name),
        cls_name=name,
        conf=conf,
        xyxy=box,
    )


def shifted(box: tuple[float, float, float, float], dx: float = 0.0, dy: float = 0.0):
    x1, y1, x2, y2 = box
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def yolo_tensor(rows: list[tuple[float, float, float, float, int, float]], num_classes: int = 14):
    """Build a ``(1, 4 + num_classes, N)`` head output from cxcywh rows.

    Each row is ``(cx, cy, w, h, class_id, score)``, which is what an exported
    YOLOv8 graph emits before decode.
    """
    tensor = np.zeros((1, 4 + num_classes, len(rows)), dtype=np.float32)
    for index, (cx, cy, w, h, cls_id, score) in enumerate(rows):
        tensor[0, :4, index] = (cx, cy, w, h)
        tensor[0, 4 + cls_id, index] = score
    return tensor
