"""Runtime execution policy for edge / NPU inference."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class ExecutionPolicy:
    """How the runtime selects backends and providers.

    Default is **ONNX-first**. Torch / Ultralytics is opt-in via ``allow_torch``
    or env ``PPE_ALLOW_TORCH=1``.
    """

    # Preferred registry keys or raw ORT names, e.g. ("qnn", "openvino", "cpu").
    providers: tuple[str, ...] = ()
    # Prefer NPU EPs only (CPU still allowed as soft fallback unless disabled).
    npu_only: bool = False
    allow_cpu_fallback: bool = True
    # Torch / Ultralytics path — off by default for edge orientation.
    allow_torch: bool = False
    # Prefer .onnx / .int8.onnx over .pt when resolving model paths.
    prefer_onnx: bool = True
    imgsz: int = 640
    conf: float = 0.25
    # Optional explicit model path.
    model: str | None = None

    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls, **overrides) -> ExecutionPolicy:
        providers_raw = os.environ.get("PPE_PROVIDERS", "").strip()
        providers = tuple(p.strip() for p in providers_raw.split(",") if p.strip())
        npu_only = _env_bool("PPE_NPU_ONLY", False)
        allow_cpu = _env_bool("PPE_ALLOW_CPU_FALLBACK", True)
        allow_torch = _env_bool("PPE_ALLOW_TORCH", False)
        prefer_onnx = _env_bool("PPE_PREFER_ONNX", True)
        imgsz = int(os.environ.get("PPE_IMGSZ", "640"))
        conf = float(os.environ.get("PPE_CONF", "0.25"))
        model = os.environ.get("PPE_WEIGHTS") or os.environ.get("PPE_MODEL")
        policy = cls(
            providers=providers,
            npu_only=npu_only,
            allow_cpu_fallback=allow_cpu,
            allow_torch=allow_torch,
            prefer_onnx=prefer_onnx,
            imgsz=imgsz,
            conf=conf,
            model=model,
        )
        for key, value in overrides.items():
            if hasattr(policy, key) and value is not None:
                setattr(policy, key, value)
        return policy


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def resolve_model_path(
    explicit: str | Path | None,
    *,
    prefer_onnx: bool = True,
    repo_root: Path | None = None,
) -> Path:
    """Resolve a model artifact: ONNX preferred, then INT8, then .pt."""
    root = repo_root or Path.cwd()
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            return path
        raise FileNotFoundError(f"Model not found: {path}")

    candidates: list[Path] = []
    onnx_first = [
        root / "models" / "best.int8.onnx",
        root / "models" / "best.onnx",
        root / "runs" / "train" / "e4_full44k" / "weights" / "best.onnx",
        root / "runs" / "train" / "e0_n" / "weights" / "best.onnx",
        root / "baselines" / "snehilsanyal_yolov8n_css" / "models" / "best.onnx",
    ]
    pt = [
        root / "runs" / "train" / "e4_full44k" / "weights" / "best.pt",
        root / "runs" / "train" / "e0_n" / "weights" / "best.pt",
        root / "baselines" / "snehilsanyal_yolov8n_css" / "models" / "best.pt",
        root / "models" / "best.pt",
    ]
    candidates = onnx_first + pt if prefer_onnx else pt + onnx_first
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        "No model found. Export ONNX first or pass --weights path/to/model.onnx|.pt"
    )


def pin_providers(keys: Sequence[str]) -> tuple[str, ...]:
    return tuple(k.strip() for k in keys if k and k.strip())
