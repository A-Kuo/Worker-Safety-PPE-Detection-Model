"""Detector backends.

Inference runs on ONNX Runtime against an NPU. That is the only path the
runtime selects on its own, and ``auto`` always resolves to it.

``onnx``
    The deployment path. ONNX Runtime bound to an NPU execution provider, with
    the numpy postprocessing in :mod:`ppe.postprocess`. No torch anywhere.
``stub``
    Replays scripted detections. Used by the tests and by ``--backend stub``
    for wiring up a deployment before any weights exist.
``ultralytics``
    A torch reference implementation, kept for diffing the ONNX path against
    on a workstation. Opt-in twice: pick it explicitly *and* set
    ``PPE_ALLOW_TORCH=1``. It is not a deployment target.
"""

from __future__ import annotations

import ast
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ppe.postprocess import decode_yolo_output, letterbox, to_input_tensor, undo_letterbox
from ppe.providers import (
    CPU_PROVIDER,
    dynamic_axes,
    requires_static_shapes,
    resolve,
    verify_binding,
)
from ppe.schema import UNIFIED_CLASS_NAMES

TORCH_OPT_IN = "PPE_ALLOW_TORCH"


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
    """Torch reference implementation. Not a deployment target.

    Kept so an ONNX export can be diffed against the checkpoint it came from.
    Constructing it requires ``PPE_ALLOW_TORCH=1`` so it cannot be reached by
    accident from a device profile.
    """

    name = "ultralytics"

    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str | None = None,
    ) -> None:
        require_torch_opt_in()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "The ultralytics backend needs `pip install -e '.[torch]'`. "
                "Deployments export to ONNX and run on an NPU instead."
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
    """ONNX Runtime bound to an NPU, with numpy letterboxing, decode, and NMS.

    The session is built from a provider policy rather than a raw provider
    list, so a host without the requested accelerator raises at construction
    instead of quietly serving CPU inference.
    """

    name = "onnx"

    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        max_detections: int = 300,
        execution: str = "npu",
        provider: str | None = None,
        provider_options: Mapping[str, str] | None = None,
    ) -> None:
        onnxruntime = _onnxruntime()

        path = Path(weights)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model not found at {path}")

        self.weights = str(path)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_detections = int(max_detections)
        self.execution = execution

        requested = resolve(execution, provider, provider_options)
        self._session = onnxruntime.InferenceSession(
            self.weights,
            providers=[name for name, _opts in requested],
            provider_options=[opts for _name, opts in requested],
        )
        self.provider = verify_binding(self._session, requested, execution)

        self._input = self._session.get_inputs()[0]
        _check_input_shape(self._input.shape, self.provider, self.weights)
        self.imgsz = _input_size(self._input.shape, imgsz)
        self._names = _names_from_metadata(self._session)

    @property
    def providers(self) -> list[str]:
        return list(self._session.get_providers())

    @property
    def on_npu(self) -> bool:
        return self.provider != CPU_PROVIDER

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
            execution=config.execution,
            provider=config.provider,
            provider_options=config.provider_options,
        )
    if kind == "ultralytics":
        return UltralyticsBackend(
            config.weights,
            conf=config.conf,
            iou=config.iou,
            imgsz=config.imgsz,
            device=config.device,
        )
    raise ValueError(f"Unknown backend {kind!r}; expected onnx, stub, or ultralytics.")


def torch_opted_in() -> bool:
    return os.environ.get(TORCH_OPT_IN, "").strip().lower() in {"1", "true", "yes", "on"}


def require_torch_opt_in() -> None:
    """Refuse the torch path unless someone deliberately turned it on.

    Inference targets an NPU through ONNX Runtime. The torch backend exists to
    check an export against its source checkpoint, which is a workstation task,
    so reaching it takes an explicit environment variable as well as an
    explicit backend choice.
    """
    if not torch_opted_in():
        raise RuntimeError(
            "The ultralytics backend is a torch reference path, not a deployment target. "
            f"Set {TORCH_OPT_IN}=1 to use it, or export to ONNX and run on an NPU:\n"
            "  python scripts/export_onnx.py --weights best.pt --imgsz 640\n"
            "  python scripts/quantize_onnx.py --model best.onnx --calibration data/calib\n"
            "  ppe image frame.jpg --weights best.int8.onnx"
        )


def _check_input_shape(shape: Sequence[object], provider: str, weights: str) -> None:
    """Reject dynamic input shapes on accelerators that cannot compile them."""
    if not requires_static_shapes(provider):
        return
    dynamic = dynamic_axes(shape)
    if not dynamic:
        return
    raise ValueError(
        f"{provider} needs a static input shape, but {weights} declares {shape} "
        f"with dynamic axes at {dynamic}. Re-export with fixed dimensions:\n"
        "  python scripts/export_onnx.py --weights best.pt --imgsz 640"
    )


def _onnxruntime():
    try:
        import onnxruntime
    except ImportError as exc:
        raise ImportError(
            "The onnx backend needs ONNX Runtime. Install the build for your accelerator "
            "(see `ppe devices`), or `pip install onnxruntime` for CPU-only development."
        ) from exc
    return onnxruntime


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
