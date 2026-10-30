"""Session factory: resolve model + policy → inference backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ppe.compliance import Detection, WorkerCompliance, associate_ppe_to_persons
from ppe.runtime.backends import InferenceBackend, build_backend
from ppe.runtime.config import ExecutionPolicy, resolve_model_path
from ppe.runtime.providers import provider_status, resolve_providers
from ppe.specialist import merge_detections, replace_classes_for
from ppe.thresholds import effective_floor, filter_detections


@dataclass
class SpecialistSlot:
    """One class-pair specialist attached to an edge session (ONNX/NPU or opt-in Torch)."""

    backend: InferenceBackend
    replace: frozenset[str]
    path: Path


@dataclass
class EdgeSession:
    """ONNX-first inference session with optional compliance association."""

    backend: InferenceBackend
    policy: ExecutionPolicy
    model_path: Path
    specialist_backends: list[SpecialistSlot] = field(default_factory=list)
    # Backward-compatible alias: first specialist backend, or None.
    specialist_backend: InferenceBackend | None = None

    def predict(self, image_bgr: np.ndarray) -> list[Detection]:
        floor = effective_floor(self.policy.conf, self.policy.class_conf)
        detections = self.backend.predict(image_bgr, conf=floor)
        for slot in self.specialist_backends:
            detections = merge_detections(
                detections,
                slot.backend.predict(image_bgr, conf=floor),
                replace=slot.replace,
            )
        return filter_detections(detections, self.policy.conf, self.policy.class_conf)

    def predict_and_comply(
        self, image_bgr: np.ndarray
    ) -> tuple[list[Detection], list[WorkerCompliance]]:
        dets = self.predict(image_bgr)
        return dets, associate_ppe_to_persons(dets)

    def info(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "policy": {
                "providers": list(self.policy.providers),
                "npu_only": self.policy.npu_only,
                "allow_torch": self.policy.allow_torch,
                "prefer_onnx": self.policy.prefer_onnx,
                "imgsz": self.policy.imgsz,
                "conf": self.policy.conf,
                "class_conf": dict(self.policy.class_conf),
                "specialist": self.policy.specialist,
                "specialists": list(self.policy.specialists),
            },
            "backend": self.backend.info(),
            "specialists": [
                {
                    "path": str(slot.path),
                    "replace": sorted(slot.replace),
                    "backend": slot.backend.info(),
                }
                for slot in self.specialist_backends
            ],
            "provider_catalog": provider_status(),
        }


def open_session(
    model: str | Path | None = None,
    *,
    policy: ExecutionPolicy | None = None,
    repo_root: Path | None = None,
) -> EdgeSession:
    """Open an edge session.

    Resolves ONNX (and INT8) before ``.pt``. Torch requires ``allow_torch``.
    Class-pair specialists (``PPE_SPECIALISTS`` / ``PPE_SPECIALIST``) attach as extra
    ONNX/NPU backends whose classes replace the main model's for that pair.
    """
    root = repo_root or _default_root()
    pol = policy or ExecutionPolicy.from_env()
    if model is not None:
        pol.model = str(model)
    path = resolve_model_path(
        pol.model,
        prefer_onnx=pol.prefer_onnx,
        repo_root=root,
    )
    # If user passed .pt but prefer_onnx and an onnx sibling exists, use it.
    if path.suffix.lower() in {".pt", ".pth"} and pol.prefer_onnx:
        sibling = path.with_suffix(".onnx")
        int8 = path.with_name(path.stem + ".int8.onnx")
        for cand in (
            int8,
            sibling,
            root / "models" / "best.int8.onnx",
            root / "models" / "best.onnx",
        ):
            if cand.is_file():
                path = cand
                break
    backend = build_backend(path, pol)
    slots = _load_specialists(pol, root)
    return EdgeSession(
        backend=backend,
        policy=pol,
        model_path=path,
        specialist_backends=slots,
        specialist_backend=slots[0].backend if slots else None,
    )


def describe_runtime() -> dict[str, Any]:
    return {
        "resolved_providers": resolve_providers(None),
        "catalog": provider_status(),
        "default_policy": {
            k: v
            for k, v in ExecutionPolicy.from_env().__dict__.items()
            if k != "extra"
        },
    }


def _load_specialists(pol: ExecutionPolicy, root: Path) -> list[SpecialistSlot]:
    paths = list(pol.specialists) or ([pol.specialist] if pol.specialist else [])
    slots: list[SpecialistSlot] = []
    for raw in paths:
        spec_path = Path(raw)
        if not spec_path.is_absolute():
            spec_path = root / spec_path
        if not spec_path.is_file():
            raise FileNotFoundError(
                f"Specialist weights not found: {spec_path} "
                "(PPE_SPECIALISTS / PPE_SPECIALIST / policy.specialists)"
            )
        # Prefer an ONNX sibling when the policy is ONNX-first (NPU path) and a .pt was given.
        resolved = spec_path
        if resolved.suffix.lower() in {".pt", ".pth"} and pol.prefer_onnx:
            for cand in (resolved.with_suffix(".onnx"), resolved.with_name(resolved.stem + ".int8.onnx")):
                if cand.is_file():
                    resolved = cand
                    break
        backend = build_backend(resolved, pol)
        replace = _replace_classes_from_backend(backend)
        if not replace:
            raise RuntimeError(
                f"Specialist {resolved} has no recognizable class names; "
                "export with Ultralytics so ONNX metadata carries names "
                "(or use a known vest/helmet 2-class checkpoint)."
            )
        slots.append(SpecialistSlot(backend=backend, replace=replace, path=resolved))
    return slots


def _replace_classes_from_backend(backend: InferenceBackend) -> frozenset[str]:
    names = getattr(backend, "class_names", None)
    if isinstance(names, dict):
        return replace_classes_for(str(v) for v in names.values())
    model = getattr(backend, "model", None)
    raw = getattr(model, "names", None) if model is not None else None
    if isinstance(raw, dict):
        return replace_classes_for(str(v) for v in raw.values())
    return frozenset()


def _default_root() -> Path:
    # src/ppe/runtime/session.py → repo root is parents[3]
    return Path(__file__).resolve().parents[3]
