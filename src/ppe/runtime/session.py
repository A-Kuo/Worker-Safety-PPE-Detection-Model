"""Session factory: resolve model + policy → inference backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ppe.compliance import Detection, WorkerCompliance, associate_ppe_to_persons
from ppe.runtime.backends import InferenceBackend, build_backend
from ppe.runtime.config import ExecutionPolicy, resolve_model_path
from ppe.runtime.providers import provider_status, resolve_providers


@dataclass
class EdgeSession:
    """ONNX-first inference session with optional compliance association."""

    backend: InferenceBackend
    policy: ExecutionPolicy
    model_path: Path

    def predict(self, image_bgr: np.ndarray) -> list[Detection]:
        return self.backend.predict(image_bgr, conf=self.policy.conf)

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
            },
            "backend": self.backend.info(),
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
    return EdgeSession(backend=backend, policy=pol, model_path=path)


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


def _default_root() -> Path:
    # src/ppe/runtime/session.py → repo root is parents[3]
    return Path(__file__).resolve().parents[3]
