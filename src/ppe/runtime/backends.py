"""Inference backends: ONNX Runtime (default) and opt-in Torch/Ultralytics."""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from ppe.compliance import Detection
from ppe.runtime.config import ExecutionPolicy
from ppe.runtime.providers import key_for_ort_name, provider_options_for, resolve_providers
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
        self.providers = resolve_providers(
            policy.providers or None,
            npu_only=policy.npu_only,
            allow_cpu_fallback=policy.allow_cpu_fallback,
        )
        if not self.providers:
            raise RuntimeError(
                "No ORT execution providers available. Install onnxruntime "
                "and/or a vendor EP package."
            )
        provider_options = [
            provider_options_for(
                key_for_ort_name(name) or "",
                openvino_device_type=policy.openvino_device_type,
            )
            for name in self.providers
        ]
        session_kwargs: dict[str, Any] = {"providers": self.providers}
        if any(provider_options):
            session_kwargs["provider_options"] = provider_options
        self.session = ort.InferenceSession(str(self.model_path), **session_kwargs)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
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
            "input": self.input_name,
            "input_shape": list(self.input_shape),
            "imgsz": self.policy.imgsz,
            "npu_only": self.policy.npu_only,
            "openvino_device_type": self.policy.openvino_device_type,
        }


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
