"""Ultralytics YOLO wrapper that emits unified detections and compliance."""

from __future__ import annotations

from typing import Any

from ppe.compliance import Detection, WorkerCompliance, associate_ppe_to_persons
from ppe.schema import (
    COMBINED_RAW_TO_UNIFIED,
    CONSTRUCTION_TO_UNIFIED,
    HHU_TO_UNIFIED,
    UNIFIED_CLASS_NAMES,
)

_RAW_TO_UNIFIED: dict[str, str] = {
    **COMBINED_RAW_TO_UNIFIED,
    **CONSTRUCTION_TO_UNIFIED,
    **HHU_TO_UNIFIED,
}


class PPEDetector:
    """Load a YOLO checkpoint and run image prediction plus worker compliance."""

    def __init__(
        self,
        weights_path: str,
        conf: float = 0.25,
        device: str | None = None,
    ) -> None:
        self.weights_path = str(weights_path)
        self.conf = conf
        self.device = device
        self._model = _yolo_cls()(self.weights_path)

    def names(self) -> dict[int, str]:
        raw = self._model.names
        return {int(k): str(v) for k, v in dict(raw).items()}

    def predict_image(self, image) -> list[Detection]:
        results = self._run(image)
        return self._to_detections(results[0])

    def predict_and_comply(self, image) -> tuple[object, list[WorkerCompliance]]:
        """Return annotated image (numpy BGR) and compliance records."""
        results = self._run(image)
        detections = self._to_detections(results[0])
        workers = associate_ppe_to_persons(detections)
        annotated = results[0].plot()
        self._draw_compliance(annotated, workers)
        return annotated, workers

    def _run(self, image) -> Any:
        kwargs: dict[str, Any] = {"conf": self.conf, "verbose": False}
        if self.device is not None:
            kwargs["device"] = self.device
        return self._model.predict(image, **kwargs)

    def _to_detections(self, result: Any) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        native_names = {int(k): str(v) for k, v in dict(result.names).items()}
        xyxy = _as_numpy(boxes.xyxy)
        confs = _as_numpy(boxes.conf)
        clss = _as_numpy(boxes.cls)
        detections: list[Detection] = []
        for box, conf, cls_id in zip(xyxy, confs, clss, strict=False):
            raw_name = native_names[int(cls_id)]
            detections.append(
                Detection(
                    cls_name=_to_unified_name(raw_name),
                    conf=float(conf),
                    xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                )
            )
        return detections

    def _draw_compliance(self, image, workers: list[WorkerCompliance]) -> None:
        cv2 = _cv2()
        for worker in workers:
            x1, y1, x2, y2 = (int(v) for v in worker.bbox)
            color = (0, 0, 220) if worker.violations else (40, 180, 40)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            origin = (x1, max(y1 - 8, 16))
            cv2.putText(
                image,
                worker.label,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )


def _to_unified_name(name: str) -> str:
    if name in UNIFIED_CLASS_NAMES:
        return name
    return _RAW_TO_UNIFIED.get(name, name)


def _yolo_cls():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "PPEDetector requires ultralytics. Install project deps first."
        ) from exc
    return YOLO


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("PPEDetector annotation requires opencv-python.") from exc
    return cv2


def _as_numpy(tensor):
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    np = _np()
    return np.asarray(tensor)


def _np():
    import numpy as np

    return np
