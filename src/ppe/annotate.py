"""Draw detections and compliance overlays on BGR frames (shared by Torch + ORT paths)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ppe.compliance import Detection, WorkerCompliance

# BGR
_COLOR_OK = (40, 180, 40)
_COLOR_BAD = (0, 0, 220)
_COLOR_BOX = (220, 180, 40)
_COLOR_NEG = (0, 90, 255)


def annotate_frame(
    image_bgr: np.ndarray,
    detections: Sequence[Detection],
    workers: Sequence[WorkerCompliance] | None = None,
    *,
    copy: bool = True,
) -> np.ndarray:
    """Return a BGR frame with PPE boxes and optional per-worker compliance labels."""
    import cv2

    out = image_bgr.copy() if copy else image_bgr
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.xyxy)
        color = _COLOR_NEG if det.cls_name.startswith("no_") else _COLOR_BOX
        if det.cls_name.lower() == "person":
            continue  # worker overlay draws the person box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det.cls_name} {det.conf:.2f}"
        cv2.putText(
            out,
            label,
            (x1, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    if workers:
        draw_compliance(out, workers, copy=False)
    return out


def draw_compliance(
    image_bgr: np.ndarray,
    workers: Sequence[WorkerCompliance],
    *,
    copy: bool = True,
) -> np.ndarray:
    """Draw person boxes and ``WorkerCompliance.label`` strings."""
    import cv2

    out = image_bgr.copy() if copy else image_bgr
    for worker in workers:
        x1, y1, x2, y2 = (int(v) for v in worker.bbox)
        color = _COLOR_BAD if (worker.missing or worker.violations) else _COLOR_OK
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            worker.label,
            (x1, max(y1 - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out
