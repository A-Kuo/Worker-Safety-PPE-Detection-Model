"""Lazy detector / edge-session construction shared by the API and Streamlit UI.

Camera and live paths prefer the ONNX Runtime edge stack (NPU → GPU → CPU).
Torch / Ultralytics is only used when the resolved weights are ``.pt`` (or
``PPE_ALLOW_TORCH=1`` forces it).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.paths import REPO_ROOT, ensure_src_on_path, resolve_weights_path

ensure_src_on_path()

IMPORT_ERROR: str | None
try:
    from ppe.compliance import Detection, WorkerCompliance, associate_ppe_to_persons
    from ppe.inference import PPEDetector
except ImportError:
    try:
        from src.ppe.compliance import Detection, WorkerCompliance, associate_ppe_to_persons
        from src.ppe.inference import PPEDetector
    except ImportError as exc:
        Detection = None  # type: ignore[misc, assignment]
        WorkerCompliance = None  # type: ignore[misc, assignment]
        associate_ppe_to_persons = None  # type: ignore[misc, assignment]
        PPEDetector = None  # type: ignore[misc, assignment]
        IMPORT_ERROR = str(exc)
    else:
        IMPORT_ERROR = None
else:
    IMPORT_ERROR = None


def default_device() -> str | None:
    env = os.environ.get("PPE_DEVICE")
    return env.strip() if env and env.strip() else None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _prefer_npu_default() -> bool:
    """Camera / demo default: prioritize NPU unless explicitly disabled."""
    return _env_bool("PPE_PREFER_NPU", True)


def _default_openvino_device(*, prefer_npu: bool) -> str | None:
    env = os.environ.get("PPE_OPENVINO_DEVICE")
    if env and env.strip():
        return env.strip()
    if prefer_npu:
        # Intel AI Boost NPU first, Arc iGPU next — ignored when OpenVINO EP is absent.
        return "AUTO:NPU,GPU"
    return None


@dataclass
class TimedResult:
    annotated: Any
    workers: list
    latency_ms: float
    detections: list | None = None


class DemoEngine:
    """Unified predict/annotate surface over EdgeSession (preferred) or PPEDetector."""

    def __init__(self, impl: Any, *, kind: str, conf: float) -> None:
        self._impl = impl
        self.kind = kind  # "onnxruntime" | "torch"
        self._conf = float(conf)

    @property
    def conf(self) -> float:
        return self._conf

    @conf.setter
    def conf(self, value: float) -> None:
        self._conf = float(value)
        if hasattr(self._impl, "conf"):
            self._impl.conf = self._conf
        policy = getattr(self._impl, "policy", None)
        if policy is not None and hasattr(policy, "conf"):
            policy.conf = self._conf

    def predict_image(self, image_bgr) -> list:
        if self.kind == "onnxruntime":
            return self._impl.predict(image_bgr)
        return self._impl.predict_image(image_bgr)

    def predict_and_comply(self, image_bgr) -> tuple[Any, list]:
        if self.kind == "onnxruntime":
            return self._impl.predict_and_annotate(image_bgr)
        return self._impl.predict_and_comply(image_bgr)

    def predict_timed(self, image_bgr) -> TimedResult:
        t0 = time.perf_counter()
        annotated, workers = self.predict_and_comply(image_bgr)
        ms = (time.perf_counter() - t0) * 1000.0
        return TimedResult(annotated=annotated, workers=workers, latency_ms=ms)

    def info(self) -> dict[str, Any]:
        base: dict[str, Any] = {"engine": self.kind, "conf": self._conf}
        if self.kind == "onnxruntime" and hasattr(self._impl, "info"):
            base.update(self._impl.info())
        else:
            base["model_path"] = getattr(self._impl, "weights_path", None)
            base["device"] = getattr(self._impl, "device", None)
        return base


def build_engine(
    weights: str | Path | None = None,
    conf: float = 0.25,
    device: str | None = None,
    class_conf: dict[str, float] | None = None,
    specialist: str | Path | None = None,
    *,
    prefer_npu: bool | None = None,
    providers: tuple[str, ...] | None = None,
    openvino_device: str | None = None,
    npu_only: bool | None = None,
) -> DemoEngine:
    """Build the demo inference engine.

    Prefer ONNX Runtime with NPU-first providers. Fall back to Torch only for
    ``.pt`` weights (implicitly allowing torch for that case).
    """
    if IMPORT_ERROR is not None and PPEDetector is None:
        raise RuntimeError(
            "ppe is not importable. Prefer `pip install -e .` or put src/ on "
            f"PYTHONPATH (src.ppe remains a fallback). Last error: {IMPORT_ERROR}"
        )

    prefer_npu = _prefer_npu_default() if prefer_npu is None else prefer_npu
    path = resolve_weights_path(str(weights) if weights else None, prefer_onnx=True)
    if not path.is_file():
        raise FileNotFoundError(
            f"Weights not found at {path}. Set PPE_WEIGHTS or pass a weights path. "
            "For NPU camera use, export ONNX first (see docs/edge_npu.md)."
        )

    if class_conf is None:
        from ppe.thresholds import parse_class_conf

        class_conf = parse_class_conf(os.environ.get("PPE_CLASS_CONF"))

    suffix = path.suffix.lower()
    use_onnx = suffix == ".onnx" or (
        suffix in {".pt", ".pth"} and _env_bool("PPE_PREFER_ONNX", True) and _onnx_sibling(path) is not None
    )
    if use_onnx and suffix in {".pt", ".pth"}:
        sibling = _onnx_sibling(path)
        assert sibling is not None
        path = sibling

    if path.suffix.lower() == ".onnx":
        return _build_edge_engine(
            path,
            conf=conf,
            class_conf=class_conf,
            specialist=specialist,
            prefer_npu=prefer_npu,
            providers=providers,
            openvino_device=openvino_device,
            npu_only=npu_only,
        )

    # Torch fallback for .pt when no ONNX sibling exists.
    return _build_torch_engine(
        path,
        conf=conf,
        device=device,
        class_conf=class_conf,
        specialist=specialist,
    )


def build_detector(
    weights: str | Path | None = None,
    conf: float = 0.25,
    device: str | None = None,
    class_conf: dict[str, float] | None = None,
    specialist: str | Path | None = None,
):
    """Backward-compatible constructor — returns a :class:`DemoEngine`."""
    return build_engine(
        weights,
        conf=conf,
        device=device,
        class_conf=class_conf,
        specialist=specialist,
    )


def runtime_status(*, prefer_npu: bool | None = None) -> dict[str, Any]:
    """Provider catalog + resolved preference for the UI /health."""
    prefer_npu = _prefer_npu_default() if prefer_npu is None else prefer_npu
    try:
        from ppe.runtime import describe_runtime, provider_status, resolve_providers
        from ppe.runtime.config import ExecutionPolicy

        env_npu = npu_only_env()
        policy = ExecutionPolicy.from_env(
            npu_only=env_npu if env_npu is not None else False,
            openvino_device_type=_default_openvino_device(prefer_npu=prefer_npu),
        )
        resolved = resolve_providers(
            policy.providers or None,
            npu_only=policy.npu_only,
            allow_cpu_fallback=policy.allow_cpu_fallback,
        )
        return {
            "prefer_npu": prefer_npu,
            "npu_only": policy.npu_only,
            "openvino_device_type": policy.openvino_device_type,
            "resolved_providers": resolved,
            "catalog": provider_status(),
            "describe": describe_runtime(),
        }
    except Exception as exc:  # noqa: BLE001 — status is best-effort for the UI
        return {"prefer_npu": prefer_npu, "error": str(exc)}


def npu_only_env() -> bool | None:
    raw = os.environ.get("PPE_NPU_ONLY")
    if raw is None or raw.strip() == "":
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _onnx_sibling(pt_path: Path) -> Path | None:
    int8 = pt_path.with_name(pt_path.stem + ".int8.onnx")
    onnx = pt_path.with_suffix(".onnx")
    for cand in (int8, onnx, REPO_ROOT / "models" / "best.int8.onnx", REPO_ROOT / "models" / "best.onnx"):
        if cand.is_file():
            return cand
    return None


def _build_edge_engine(
    path: Path,
    *,
    conf: float,
    class_conf: dict[str, float] | None,
    specialist: str | Path | None,
    prefer_npu: bool,
    providers: tuple[str, ...] | None,
    openvino_device: str | None,
    npu_only: bool | None,
) -> DemoEngine:
    from ppe.runtime import ExecutionPolicy, open_session

    env_npu = npu_only_env()
    # prefer_npu only biases OpenVINO device targeting + EP auto-order (already NPU→GPU→CPU).
    # npu_only is opt-in so CUDA hosts are not forced onto CPU when no NPU EP is installed.
    if npu_only is not None:
        effective_npu_only = npu_only
    elif env_npu is not None:
        effective_npu_only = env_npu
    else:
        effective_npu_only = False
    ov_device = openvino_device or _default_openvino_device(prefer_npu=prefer_npu)
    provider_tuple = providers
    if provider_tuple is None:
        raw = os.environ.get("PPE_PROVIDERS", "").strip()
        provider_tuple = tuple(p.strip() for p in raw.split(",") if p.strip()) if raw else ()

    spec = specialist or os.environ.get("PPE_SPECIALIST")
    policy = ExecutionPolicy.from_env(
        providers=provider_tuple,
        npu_only=effective_npu_only,
        conf=float(conf),
        class_conf=class_conf or {},
        model=str(path),
        specialist=str(spec) if spec else None,
        openvino_device_type=ov_device,
        allow_torch=False,
        prefer_onnx=True,
    )
    session = open_session(path, policy=policy, repo_root=REPO_ROOT)
    return DemoEngine(session, kind="onnxruntime", conf=conf)


def _build_torch_engine(
    path: Path,
    *,
    conf: float,
    device: str | None,
    class_conf: dict[str, float] | None,
    specialist: str | Path | None,
) -> DemoEngine:
    if PPEDetector is None:
        raise RuntimeError(
            f"Torch weights at {path} require ultralytics/ppe. "
            "Export ONNX for the NPU path, or install project deps."
        )
    kwargs: dict = {"weights_path": str(path), "conf": float(conf), "class_conf": class_conf}
    spec = specialist or os.environ.get("PPE_SPECIALIST")
    if spec:
        if not Path(spec).is_file():
            raise FileNotFoundError(f"Vest specialist weights not found: {spec} (PPE_SPECIALIST)")
        kwargs["specialist_weights"] = str(spec)
    chosen = device if device is not None else default_device()
    if chosen:
        kwargs["device"] = chosen
    return DemoEngine(PPEDetector(**kwargs), kind="torch", conf=conf)
