"""Inference backends: ONNX Runtime (default) and opt-in Torch/Ultralytics."""

from __future__ import annotations

import ast
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from ppe.compliance import Detection
from ppe.runtime.config import ExecutionPolicy
from ppe.runtime.fingerprint import runtime_fingerprint
from ppe.runtime.providers import (
    key_for_ort_name,
    prepare_ort_environment,
    provider_options_for,
    require_npu_providers,
    resolve_providers,
)
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


class InferenceBackend(ABC):
    name: str

    @abstractmethod
    def predict(self, image_bgr: np.ndarray, *, conf: float) -> list[Detection]:
        raise NotImplementedError

    @abstractmethod
    def info(self) -> dict[str, Any]:
        raise NotImplementedError


class OnnxRuntimeBackend(InferenceBackend):
    """NPU/edge path: ONNX model via ORT execution providers."""

    name = "onnxruntime"

    def __init__(
        self,
        model_path: Path,
        policy: ExecutionPolicy,
        class_names: dict[int, str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for edge inference. "
                "pip install onnxruntime  (or onnxruntime-gpu / vendor EP builds)"
            ) from exc

        self.model_path = Path(model_path)
        self.policy = policy
        self.session, self.providers, self.providers_missing = open_ort_session(
            ort, self.model_path, policy
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.static_input = is_static_shape(self.input_shape)
        if policy.strict_npu and not self.static_input:
            raise RuntimeError(
                f"Strict NPU mode requires a static input shape; {self.model_path.name} has "
                f"{list(self.input_shape)}. Export with `ppe export` (static 640x640 by default)."
            )
        self.fingerprint = runtime_fingerprint(self.model_path)
        # Prefer the class names Ultralytics embeds in the ONNX metadata. Falling back to
        # UNIFIED_CLASS_NAMES *by index* is only right for a 14-class model already in unified
        # order; a 2-class specialist (vest, no_vest) would otherwise decode as helmet/no_helmet.
        self.class_names = class_names or _onnx_metadata_names(self.session) or {
            i: name for i, name in enumerate(UNIFIED_CLASS_NAMES)
        }

    def predict(self, image_bgr: np.ndarray, *, conf: float) -> list[Detection]:
        tensor, meta = _letterbox(image_bgr, self.policy.imgsz)
        outputs = self.session.run(None, {self.input_name: tensor})
        return _decode_yolo_onnx(outputs, meta, conf=conf, names=self.class_names)

    def info(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "model": str(self.model_path),
            "providers_requested": list(self.policy.providers),
            "providers_active": list(self.session.get_providers()),
            "providers_missing": list(self.providers_missing),
            "input": self.input_name,
            "input_shape": list(self.input_shape),
            "imgsz": self.policy.imgsz,
            "npu_only": self.policy.effective_npu_only,
            "strict_npu": self.policy.strict_npu,
            "static_input": self.static_input,
            "openvino_device_type": self.policy.effective_openvino_device,
            "class_names": dict(self.class_names),
            "fingerprint": self.fingerprint,
        }


def openvino_device_is_npu(device_type: str | None) -> bool:
    """True only for a device string that targets the NPU alone (``NPU`` or ``NPU.<n>``)."""
    d = (device_type or "").strip().upper()
    return d == "NPU" or (d.startswith("NPU.") and d[4:].isdigit())


def is_static_shape(shape: Any) -> bool:
    """True when every input dim is a concrete positive int (NPUs need this)."""
    return all(isinstance(d, int) and d > 0 for d in shape)


def plan_providers(policy: ExecutionPolicy) -> tuple[list[str], list[dict]]:
    """Resolve the ORT provider list and per-provider options for ``policy``.

    Single source of truth shared by the backend and ``ppe verify-npu`` so the placement report
    describes exactly the session inference will run.
    """
    providers = resolve_providers(
        policy.providers or None,
        npu_only=policy.effective_npu_only,
        allow_cpu_fallback=policy.effective_allow_cpu_fallback,
        exclude_cpu=policy.strict_npu,
    )
    require_npu_providers(
        providers,
        npu_only=policy.effective_npu_only,
        allow_cpu_fallback=policy.effective_allow_cpu_fallback,
    )
    if not providers:
        raise RuntimeError(
            "No ORT execution providers available. Install onnxruntime "
            "and/or a vendor EP package"
            + (" (NPU EP required — CPU fallback is disabled)." if policy.effective_npu_only else ".")
        )
    if policy.strict_npu and "OpenVINOExecutionProvider" in providers:
        device = policy.effective_openvino_device or ""
        if not openvino_device_is_npu(device):
            raise RuntimeError(
                f"Strict NPU mode needs the OpenVINO device pinned to the NPU, got {device!r}. "
                "AUTO/HETERO/GPU/CPU may place the model on the iGPU or CPU. Use "
                "--openvino-device NPU, or drop --strict-npu to benchmark other devices."
            )
    options = [
        provider_options_for(
            key_for_ort_name(name) or "",
            openvino_device_type=policy.effective_openvino_device,
            openvino_cache_dir=policy.openvino_cache_dir,
            qnn_backend_path=policy.qnn_backend_path,
        )
        for name in providers
    ]
    return providers, options


def make_session_options(
    ort: Any, policy: ExecutionPolicy, *, profile_prefix: str | None = None
) -> Any:
    """``SessionOptions`` for ``policy``. Strict mode forbids silent CPU execution of any node."""
    so = ort.SessionOptions()
    if policy.strict_npu:
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if profile_prefix:
        so.enable_profiling = True
        so.profile_file_prefix = profile_prefix
    return so


def open_ort_session(
    ort: Any,
    model_path: Path,
    policy: ExecutionPolicy,
    *,
    profile_prefix: str | None = None,
) -> tuple[Any, list[str], list[str]]:
    """Create the ORT session for ``policy`` → ``(session, providers_requested, providers_missing)``.

    One factory for inference and for ``ppe verify-npu`` so both describe the same session. Guards
    against two ways ORT silently degrades to the CPU:
    - a node the NPU EP rejects (strict mode forbids it via ``disable_cpu_ep_fallback``), and
    - an EP that fails to *load* (e.g. missing ``openvino.dll``): ORT only logs a warning and runs on
      the CPU. A requested provider absent from ``session.get_providers()`` is an error in strict mode
      and a ``RuntimeWarning`` otherwise.
    """
    providers, options = plan_providers(policy)
    prepare_ort_environment()
    kwargs: dict[str, Any] = {
        "providers": providers,
        "sess_options": make_session_options(ort, policy, profile_prefix=profile_prefix),
    }
    if any(options):
        kwargs["provider_options"] = options
    try:
        session = ort.InferenceSession(str(model_path), **kwargs)
    except Exception as exc:
        if not policy.strict_npu:
            raise
        raise RuntimeError(
            f"Strict NPU mode: {Path(model_path).name} could not be placed entirely on {providers}. "
            "Either an op is unsupported by the NPU EP, or the EP/device failed to load (check the "
            "ONNX Runtime log above, and for OpenVINO that the matching `openvino` runtime package is "
            "installed). Run `ppe verify-npu --soft` to list CPU-resident ops. "
            f"ORT said: {exc}"
        ) from exc
    active = list(session.get_providers())
    missing = [p for p in providers if p not in active]
    if missing:
        msg = (
            f"Requested execution provider(s) {missing} did not load; active={active}. The model "
            "will not run where you asked. For OpenVINO install the matching `openvino` runtime "
            "(pip install -e '.[edge-intel-npu]')."
        )
        if policy.strict_npu:
            raise RuntimeError("Strict NPU mode: " + msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return session, providers, missing


class TorchBackend(InferenceBackend):
    """Opt-in Ultralytics / Torch path. Not used for NPU-only deployments."""

    name = "torch"

    def __init__(self, model_path: Path, policy: ExecutionPolicy) -> None:
        if not policy.allow_torch:
            raise RuntimeError(
                "Torch backend is gated. Pass allow_torch=True or set PPE_ALLOW_TORCH=1."
            )
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        self.policy = policy
        self.model = YOLO(str(self.model_path))
        # Same shape as OnnxRuntimeBackend.class_names so specialist replace-sets resolve uniformly.
        self.class_names = {int(k): str(v) for k, v in dict(self.model.names).items()}

    def predict(self, image_bgr: np.ndarray, *, conf: float) -> list[Detection]:
        results = self.model.predict(
            image_bgr,
            conf=conf,
            imgsz=self.policy.imgsz,
            verbose=False,
        )
        return _detections_from_ultralytics(results[0])

    def info(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "model": str(self.model_path),
            "imgsz": self.policy.imgsz,
            "gated": True,
            "class_names": dict(self.class_names),
        }


def build_backend(model_path: Path, policy: ExecutionPolicy) -> InferenceBackend:
    """Select backend from model suffix and policy.

    ONNX / INT8 → ORT. ``.pt`` only when ``allow_torch``.
    """
    suffix = model_path.suffix.lower()
    if suffix in {".onnx"}:
        return OnnxRuntimeBackend(model_path, policy)
    if suffix in {".pt", ".pth"}:
        if not policy.allow_torch:
            raise RuntimeError(
                f"Refusing Torch weights {model_path} while allow_torch=False. "
                "Export ONNX first, or set PPE_ALLOW_TORCH=1 for opt-in."
            )
        return TorchBackend(model_path, policy)
    raise ValueError(f"Unsupported model type: {model_path}")


def _letterbox(image_bgr: np.ndarray, imgsz: int) -> tuple[np.ndarray, dict]:
    import cv2

    h0, w0 = image_bgr.shape[:2]
    scale = min(imgsz / h0, imgsz / w0)
    nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
    resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    top = (imgsz - nh) // 2
    left = (imgsz - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
    meta = {"scale": scale, "pad": (left, top), "orig": (w0, h0), "imgsz": imgsz}
    return tensor, meta


def _decode_yolo_onnx(
    outputs: list[Any],
    meta: dict,
    *,
    conf: float,
    names: dict[int, str],
) -> list[Detection]:
    """Decode common Ultralytics YOLOv8 ONNX output shapes."""
    out = np.asarray(outputs[0])
    # (1, 4+nc, N) or (1, N, 4+nc)
    if out.ndim != 3:
        out = out.reshape(1, out.shape[-2], out.shape[-1])
    if out.shape[1] < out.shape[2] and out.shape[1] <= 128:
        # (1, 4+nc, N) → (N, 4+nc)
        pred = out[0].transpose(1, 0)
    else:
        pred = out[0]

    if pred.shape[1] < 5:
        return []

    boxes = pred[:, :4]
    scores = pred[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confs = scores[np.arange(scores.shape[0]), class_ids]
    keep = confs >= conf
    boxes, confs, class_ids = boxes[keep], confs[keep], class_ids[keep]
    if len(boxes) == 0:
        return []

    # xywh → xyxy in letterboxed space, then undo pad/scale
    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    left, top = meta["pad"]
    scale = meta["scale"]
    w0, h0 = meta["orig"]
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - left) / scale
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - top) / scale
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w0)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h0)

    # Simple class-agnostic NMS
    order = _nms(xyxy, confs, iou_thresh=0.45)
    detections: list[Detection] = []
    for i in order:
        cid = int(class_ids[i])
        raw = names.get(cid, str(cid))
        detections.append(
            Detection(
                cls_name=_to_unified(raw),
                conf=float(confs[i]),
                xyxy=(float(xyxy[i, 0]), float(xyxy[i, 1]), float(xyxy[i, 2]), float(xyxy[i, 3])),
            )
        )
    return detections


def _onnx_metadata_names(session: Any) -> dict[int, str] | None:
    """Class names from the ONNX ``names`` metadata Ultralytics writes, or None if absent/unparseable."""
    try:
        raw = session.get_modelmeta().custom_metadata_map.get("names")
        if not raw:
            return None
        parsed = ast.literal_eval(raw)
        return {int(k): str(v) for k, v in parsed.items()}
    except Exception:  # noqa: BLE001 - metadata is optional; fall back to the positional default
        return None


def _nms(xyxy: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou <= iou_thresh]
    return keep


def _detections_from_ultralytics(result: Any) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    native = {int(k): str(v) for k, v in dict(result.names).items()}
    xyxy = boxes.xyxy
    confs = boxes.conf
    clss = boxes.cls
    if hasattr(xyxy, "cpu"):
        xyxy, confs, clss = xyxy.cpu().numpy(), confs.cpu().numpy(), clss.cpu().numpy()
    else:
        xyxy, confs, clss = np.asarray(xyxy), np.asarray(confs), np.asarray(clss)
    out: list[Detection] = []
    for box, conf, cls_id in zip(xyxy, confs, clss, strict=False):
        out.append(
            Detection(
                cls_name=_to_unified(native[int(cls_id)]),
                conf=float(conf),
                xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            )
        )
    return out


def _to_unified(name: str) -> str:
    if name in UNIFIED_CLASS_NAMES:
        return name
    return _RAW_TO_UNIFIED.get(name, name)
