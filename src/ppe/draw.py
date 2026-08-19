"""Frame annotation.

Drawing lives here rather than behind ``ultralytics``'s plotting helper so the
ONNX path renders identically to the torch path.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

BOX_COLOR = (200, 170, 60)
COMPLIANT_COLOR = (60, 175, 60)
VIOLATION_COLOR = (50, 50, 210)
TEXT_COLOR = (255, 255, 255)


def annotate(
    frame: np.ndarray,
    detections: Sequence,
    workers: Sequence = (),
    show_conf: bool = True,
) -> np.ndarray:
    """Draw detections and per-worker compliance onto a copy of ``frame``."""
    cv2 = _cv2()
    canvas = frame.copy()
    scale = _text_scale(canvas)

    for det in detections:
        if det.cls_name.lower() == "person":
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in det.xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), BOX_COLOR, 1)
        text = f"{det.cls_name} {det.conf:.2f}" if show_conf else det.cls_name
        _label(canvas, text, (x1, y1), BOX_COLOR, scale)

    for worker in workers:
        color = VIOLATION_COLOR if worker.violations else COMPLIANT_COLOR
        x1, y1, x2, y2 = (int(round(v)) for v in worker.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        _label(canvas, worker.label, (x1, y1), color, scale)

    return canvas


def draw_banner(frame: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    """Write a few status lines into the top-left corner."""
    cv2 = _cv2()
    canvas = frame if frame.flags.writeable else frame.copy()
    scale = _text_scale(canvas)
    y = int(18 * scale / 0.5)
    for line in lines:
        cv2.putText(
            canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, TEXT_COLOR, 1, cv2.LINE_AA
        )
        y += int(20 * scale / 0.5)
    return canvas


def _label(canvas, text: str, origin: tuple[int, int], color, scale: float) -> None:
    cv2 = _cv2()
    x, y = origin
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    top = max(y - height - baseline - 2, 0)
    cv2.rectangle(canvas, (x, top), (x + width + 4, top + height + baseline + 2), color, -1)
    cv2.putText(
        canvas,
        text,
        (x + 2, top + height + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )


def _text_scale(frame: np.ndarray) -> float:
    return max(0.4, min(0.9, frame.shape[1] / 1600.0))


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Annotation needs opencv (pip install opencv-python-headless).") from exc
    return cv2
