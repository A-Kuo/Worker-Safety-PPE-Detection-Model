"""Ultralytics YOLO wrapper that emits unified detections and compliance."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ppe.compliance import Detection, WorkerCompliance, associate_ppe_to_persons
from ppe.schema import (
    COMBINED_RAW_TO_UNIFIED,
    CONSTRUCTION_TO_UNIFIED,
    HHU_TO_UNIFIED,
    UNIFIED_CLASS_NAMES,
)
from ppe.specialist import SPECIALIST_CLASSES
from ppe.thresholds import effective_floor

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
        class_conf: Mapping[str, float] | None = None,
        specialist_weights: str | None = None,
    ) -> None:
        self.weights_path = str(weights_path)
        self.conf = conf
        # Per-class overrides of ``conf`` (see ppe.thresholds), keyed by unified class name.
        self.class_conf = dict(class_conf or {})
        self.device = device
        self._model = _yolo_cls()(self.weights_path)
        # Optional 2-class vest/no_vest model whose output replaces the main model's for those classes.
        self._specialist = _yolo_cls()(str(specialist_weights)) if specialist_weights else None

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
        kwargs: dict[str, Any] = {"conf": effective_floor(self.conf, self.class_conf), "verbose": False}
        if self.device is not None:
            kwargs["device"] = self.device
        results = self._model.predict(image, **kwargs)
        if self._specialist is not None:
            for main, spec in zip(results, self._specialist.predict(image, **kwargs), strict=True):
                self._merge_specialist(main, spec)
        if self.class_conf:
            for result in results:
                self._apply_class_conf(result)
        return results

    def _merge_specialist(self, main: Any, spec: Any) -> None:
        """Replace the main result's vest/no_vest boxes with the specialist's, in place."""
        merged = _merge_specialist_rows(main, spec)
        if merged is None:
            return
        from ultralytics.engine.results import Boxes

        main.boxes = Boxes(merged, main.orig_shape)

    def _apply_class_conf(self, result: Any) -> None:
        """Drop boxes below their class's threshold, in place, so plot() and detections agree."""
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return
        native_names = {int(k): str(v) for k, v in dict(result.names).items()}
        confs = _as_numpy(boxes.conf)
        clss = _as_numpy(boxes.cls)
        keep = _np().asarray(
            [
                float(c) >= self.class_conf.get(_to_unified_name(native_names[int(k)]), self.conf)
                for c, k in zip(confs, clss, strict=False)
            ],
            dtype=bool,
        )
        if not keep.all():
            result.boxes = boxes[keep]

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


def _merge_specialist_rows(main: Any, spec: Any):
    """(N, 6) rows [x1, y1, x2, y2, conf, main_class_id]; None if the main model has no vest classes."""
    np = _np()
    main_names = {int(k): str(v) for k, v in dict(main.names).items()}
    id_for: dict[str, int] = {}
    for cid, raw in main_names.items():
        id_for.setdefault(_to_unified_name(raw), cid)
    if not SPECIALIST_CLASSES.intersection(id_for):
        return None
    spec_names = {int(k): _to_unified_name(str(v)) for k, v in dict(spec.names).items()}

    rows = []
    if getattr(main, "boxes", None) is not None and len(main.boxes):
        for row in _as_numpy(main.boxes.data):
            if _to_unified_name(main_names[int(row[5])]) not in SPECIALIST_CLASSES:
                rows.append([float(v) for v in row[:6]])
    if getattr(spec, "boxes", None) is not None and len(spec.boxes):
        for row in _as_numpy(spec.boxes.data):
            name = spec_names.get(int(row[5]))
            if name in SPECIALIST_CLASSES and name in id_for:
                rows.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(id_for[name])])
    if not rows:
        return np.zeros((0, 6), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


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
