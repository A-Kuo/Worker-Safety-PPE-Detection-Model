"""Ultralytics YOLO wrapper that emits unified detections and compliance.

This is the single entry point everything else (scripts, FastAPI, Streamlit)
uses to run the model. It hides three things callers shouldn't have to deal
with themselves:

1. **Ultralytics/OpenCV are optional imports.** They're only pulled in the
   moment a :class:`PPEDetector` is constructed or an image is annotated —
   see :func:`_yolo_cls` / :func:`_cv2` — so `ppe.schema` / `ppe.compliance`
   stay importable in environments (like CI) without heavy ML deps.
2. **Class-name normalization.** A checkpoint trained on raw Combined /
   Construction / HHU labels reports *raw* class names; everything past
   this module should only ever see unified names (see `_to_unified_name`).
3. **Tensor plumbing.** Ultralytics box tensors can be torch (CPU or CUDA)
   or already-numpy; `_as_numpy` normalizes that once instead of forcing
   every caller to think about `.detach().cpu().numpy()`.
"""

from __future__ import annotations

from typing import Any

from ppe.compliance import Detection, WorkerCompliance, associate_ppe_to_persons
from ppe.schema import (
    COMBINED_RAW_TO_UNIFIED,
    CONSTRUCTION_TO_UNIFIED,
    HHU_TO_UNIFIED,
    UNIFIED_CLASS_NAMES,
)

# Merged lookup for "whatever raw name a checkpoint's .names dict reports" ->
# unified name. Safe to merge blindly: the three source mappings don't
# collide on keys that map to different unified classes (e.g. "Person" only
# ever means "person" in every dataset).
_RAW_TO_UNIFIED: dict[str, str] = {
    **COMBINED_RAW_TO_UNIFIED,
    **CONSTRUCTION_TO_UNIFIED,
    **HHU_TO_UNIFIED,
}


class PPEDetector:
    """Load a YOLO checkpoint and run image prediction plus worker compliance.

    This is intentionally a thin wrapper: all "what does missing PPE mean"
    logic lives in :mod:`ppe.compliance`, and all "how do label ids map to
    names" logic lives in :mod:`ppe.schema`. This class's only job is to run
    the model and translate its output into unified :class:`Detection`
    objects.
    """

    def __init__(
        self,
        weights_path: str,
        conf: float = 0.25,
        device: str | None = None,
    ) -> None:
        """Load ``weights_path`` immediately (eager, not lazy).

        ``conf`` is the confidence threshold applied on every predict call
        and can be changed later via ``detector.conf = ...`` (the FastAPI
        app does this to reuse one cached model across requests with
        different thresholds). ``device`` follows Ultralytics conventions
        (``"cpu"``, ``"cuda"``, ``"0"``, ...); leave it ``None`` to let
        Ultralytics auto-select.
        """
        self.weights_path = str(weights_path)
        self.conf = conf
        self.device = device
        self._model = _yolo_cls()(self.weights_path)

    def names(self) -> dict[int, str]:
        """Return the checkpoint's native ``{class_id: raw_name}`` mapping.

        These are the *raw* names baked into the checkpoint at train time,
        not unified names — useful for debugging which dataset a given
        weights file was trained on.
        """
        raw = self._model.names
        return {int(k): str(v) for k, v in dict(raw).items()}

    def predict_image(self, image) -> list[Detection]:
        """Run detection on one image and return unified :class:`Detection` boxes.

        ``image`` accepts anything Ultralytics' ``model.predict`` accepts:
        a file path/URL, a numpy array (BGR), or a PIL image.
        """
        results = self._run(image)
        return self._to_detections(results[0])

    def predict_and_comply(self, image) -> tuple[object, list[WorkerCompliance]]:
        """Detect, associate PPE to persons, and draw the verdict on the frame.

        Returns ``(annotated_image, workers)`` where ``annotated_image`` is
        a numpy BGR array (Ultralytics' own box/label plot, with our
        per-worker compliance boxes and labels drawn on top — green for
        compliant, red for any violation) and ``workers`` is the same list
        of :class:`WorkerCompliance` records used to draw it, so callers get
        structured data and a visual for free from one model pass.
        """
        results = self._run(image)
        detections = self._to_detections(results[0])
        workers = associate_ppe_to_persons(detections)
        annotated = results[0].plot()
        self._draw_compliance(annotated, workers)
        return annotated, workers

    def _run(self, image) -> Any:
        """Call the underlying Ultralytics model with this detector's settings."""
        kwargs: dict[str, Any] = {"conf": self.conf, "verbose": False}
        if self.device is not None:
            kwargs["device"] = self.device
        return self._model.predict(image, **kwargs)

    def _to_detections(self, result: Any) -> list[Detection]:
        """Convert one Ultralytics ``Results`` object into unified :class:`Detection` boxes."""
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
        """Draw one box + label per worker directly onto ``image`` (mutated in place)."""
        cv2 = _cv2()
        for worker in workers:
            x1, y1, x2, y2 = (int(v) for v in worker.bbox)
            color = (0, 0, 220) if worker.violations else (40, 180, 40)  # BGR: red / green
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            origin = (x1, max(y1 - 8, 16))  # keep the label on-screen even near the top edge
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
    """Map a checkpoint's raw class name to the unified schema.

    Names already in the unified schema pass through unchanged (so this is
    safe to call on output from a checkpoint trained directly on unified
    labels). Anything not recognized in either form is returned as-is
    rather than raised — an unexpected class name shouldn't crash inference,
    it just won't be treated as wearable PPE by the compliance layer.
    """
    if name in UNIFIED_CLASS_NAMES:
        return name
    return _RAW_TO_UNIFIED.get(name, name)


def _yolo_cls():
    """Import Ultralytics' ``YOLO`` class lazily, with a clearer error if it's missing."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "PPEDetector requires ultralytics. Install project deps first."
        ) from exc
    return YOLO


def _cv2():
    """Import OpenCV lazily, with a clearer error if it's missing."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("PPEDetector annotation requires opencv-python.") from exc
    return cv2


def _as_numpy(tensor):
    """Normalize a torch tensor (CPU or CUDA) or numpy array into plain numpy.

    Ultralytics box tensors are torch by default. This detaches from the
    autograd graph, moves off any GPU, and converts to numpy — each step is
    a no-op if the object doesn't have that method, so it works for both
    torch tensors and inputs that are already numpy (e.g. in tests).
    """
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    np = _np()
    return np.asarray(tensor)


def _np():
    """Import numpy lazily to match the rest of this module's optional-import style."""
    import numpy as np

    return np
