"""Detector backends.

Three implementations share one interface so the rest of the pipeline does not
care what is behind it:

``ultralytics``
    The PyTorch path. Convenient on a workstation, heavy on a device.
``onnx``
    ONNX Runtime plus the numpy postprocessing in :mod:`ppe.postprocess`. This
    is the path meant for edge hardware, where torch is usually not installed.
``stub``
    Replays scripted detections. Used by the tests and by ``--backend stub``
    for wiring up a deployment before any weights exist.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ppe.postprocess import decode_yolo_output, letterbox, to_input_tensor, undo_letterbox
from ppe.schema import UNIFIED_CLASS_NAMES


@dataclass(frozen=True)
class RawDetection:
    """One box as the model reported it, before label unification."""

    cls_id: int
    cls_name: str
    conf: float
    xyxy: tuple[float, float, float, float]


@runtime_checkable
class DetectorBackend(Protocol):
    """Anything that turns a BGR frame into boxes."""

    name: str

    def class_names(self) -> dict[int, str]: ...

    def infer(self, image: np.ndarray) -> list[RawDetection]: ...

    def close(self) -> None: ...


class UltralyticsBackend:
    """YOLO checkpoint loaded through the ultralytics package."""

    name = "ultralytics"

    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "The ultralytics backend needs `pip install ultralytics`. "
                "Export to ONNX and use --backend onnx for a torch-free device."
            ) from exc
        self.weights = str(weights)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = device
        self._model = YOLO(self.weights)

    def class_names(self) -> dict[int, str]:
        return {int(k): str(v) for k, v in dict(self._model.names).items()}

    def infer(self, image: np.ndarray) -> list[RawDetection]:
        kwargs = {
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device
        result = self._model.predict(image, **kwargs)[0]
        names = {int(k): str(v) for k, v in dict(result.names).items()}
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = _to_numpy(boxes.xyxy)
        confs = _to_numpy(boxes.conf)
        classes = _to_numpy(boxes.cls)
        return _pack(xyxy, confs, classes, names)

    def close(self) -> None:
        self._model = None


class OnnxBackend:
    """ONNX Runtime session with numpy letterboxing, decode, and NMS."""

    name = "onnx"

    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        max_detections: int = 300,
        providers: Sequence[str] | None = None,
    ) -> None:
        try:
            import onnxruntime
        except ImportError as exc:
            raise ImportError("The onnx backend needs `pip install onnxruntime`.") from exc

        path = Path(weights)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model not found at {path}")

        self.weights = str(path)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_detections = int(max_detections)
        self._session = onnxruntime.InferenceSession(
            self.weights,
            providers=list(providers) if providers else onnxruntime.get_available_providers(),
        )
        self._input = self._session.get_inputs()[0]
        self.imgsz = _input_size(self._input.shape, imgsz)
        self._names = _names_from_metadata(self._session)

    @property
    def providers(self) -> list[str]:
        return list(self._session.get_providers())

    def class_names(self) -> dict[int, str]:
        return dict(self._names)

    def infer(self, image: np.ndarray) -> list[RawDetection]:
        canvas, info = letterbox(image, self.imgsz)
        outputs = self._session.run(None, {self._input.name: to_input_tensor(canvas)})
        boxes, scores, class_ids = decode_yolo_output(
            outputs[0],
            conf_threshold=self.conf,
            iou_threshold=self.iou,
            max_detections=self.max_detections,
            num_classes=len(self._names),
        )
        return _pack(undo_letterbox(boxes, info), scores, class_ids, self._names)

    def close(self) -> None:
        self._session = None


class StubBackend:
    """Replays a fixed list of detections, one entry per call."""

    name = "stub"

    def __init__(
        self,
        frames: Iterable[Sequence[RawDetection]] | None = None,
        names: dict[int, str] | None = None,
        loop: bool = True,
    ) -> None:
        self._frames = [list(frame) for frame in (frames or [[]])]
        self._names = names or dict(enumerate(UNIFIED_CLASS_NAMES))
        self._loop = loop
        self.calls = 0

    def class_names(self) -> dict[int, str]:
        return dict(self._names)

    def infer(self, image: np.ndarray) -> list[RawDetection]:
        index = self.calls
        self.calls += 1
        if index >= len(self._frames):
            if not self._loop:
                return []
            index %= len(self._frames)
        return list(self._frames[index])

    def close(self) -> None:
        self.calls = 0


def load_backend(config) -> DetectorBackend:
    """Build the backend named by ``config.backend_name``."""
    kind = config.backend_name
    if kind == "stub":
        return StubBackend()
    if config.weights is None:
        raise ValueError(f"The {kind} backend needs a weights path (set PPE_WEIGHTS).")
    if kind == "onnx":
        return OnnxBackend(
            config.weights,
            conf=config.conf,
            iou=config.iou,
            imgsz=config.imgsz,
            max_detections=config.max_detections,
        )
    if kind == "ultralytics":
        return UltralyticsBackend(
            config.weights,
            conf=config.conf,
            iou=config.iou,
            imgsz=config.imgsz,
            device=config.device,
        )
    raise ValueError(f"Unknown backend {kind!r}; expected ultralytics, onnx, or stub.")


def _pack(boxes, confs, classes, names: dict[int, str]) -> list[RawDetection]:
    out: list[RawDetection] = []
    for box, conf, cls_id in zip(boxes, confs, classes, strict=False):
        index = int(cls_id)
        out.append(
            RawDetection(
                cls_id=index,
                cls_name=names.get(index, str(index)),
                conf=float(conf),
                xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            )
        )
    return out


def _input_size(shape: Sequence[object], fallback: int) -> int:
    height = shape[2] if len(shape) > 2 else None
    return int(height) if isinstance(height, int) and height > 0 else int(fallback)


def _names_from_metadata(session) -> dict[int, str]:
    """Read the ``names`` map ultralytics writes into exported ONNX models."""
    raw = session.get_modelmeta().custom_metadata_map.get("names")
    if raw:
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, dict):
            return {int(k): str(v) for k, v in parsed.items()}
        if isinstance(parsed, list):
            return {i: str(v) for i, v in enumerate(parsed)}
    return dict(enumerate(UNIFIED_CLASS_NAMES))


def _to_numpy(tensor):
    for attr in ("detach", "cpu"):
        if hasattr(tensor, attr):
            tensor = getattr(tensor, attr)()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    return np.asarray(tensor)
