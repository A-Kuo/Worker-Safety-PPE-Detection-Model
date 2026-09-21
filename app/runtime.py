"""Lazy PPEDetector construction shared by the API and Streamlit UI."""

from __future__ import annotations

import os
from pathlib import Path

from app.paths import ensure_src_on_path, resolve_weights_path

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


def build_detector(
    weights: str | Path | None = None,
    conf: float = 0.25,
    device: str | None = None,
    class_conf: dict[str, float] | None = None,
    specialist: str | Path | None = None,
):
    if PPEDetector is None:
        raise RuntimeError(
            "ppe is not importable. Prefer `pip install -e .` or put src/ on "
            f"PYTHONPATH (src.ppe remains a fallback). Last error: {IMPORT_ERROR}"
        )
    path = resolve_weights_path(str(weights) if weights else None)
    if not path.is_file():
        raise FileNotFoundError(
            f"Weights not found at {path}. Set PPE_WEIGHTS or pass a weights path."
        )
    if class_conf is None:
        from ppe.thresholds import parse_class_conf

        class_conf = parse_class_conf(os.environ.get("PPE_CLASS_CONF"))
    kwargs: dict = {"weights_path": str(path), "conf": float(conf), "class_conf": class_conf}
    spec = specialist or os.environ.get("PPE_SPECIALIST")
    if spec:
        if not Path(spec).is_file():
            raise FileNotFoundError(f"Vest specialist weights not found: {spec} (PPE_SPECIALIST)")
        kwargs["specialist_weights"] = str(spec)
    chosen = device if device is not None else default_device()
    if chosen:
        kwargs["device"] = chosen
    return PPEDetector(**kwargs)
