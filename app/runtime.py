"""Lazy PPEDetector construction shared by the API and Streamlit UI.

Both `app/api/main.py` and `app/ui/streamlit_app.py` need to build a
:class:`ppe.inference.PPEDetector` the same way — resolving a weights path,
picking a device, and failing gracefully (not with a crash) if `ppe` isn't
importable or the weights file is missing. This module is the one place
that logic lives, so `/health` can report *why* the service isn't ready
instead of just that it isn't.
"""

from __future__ import annotations

import os
from pathlib import Path

from app.paths import ensure_src_on_path, resolve_weights_path

ensure_src_on_path()

# Two-level import fallback: prefer `ppe` (src/ on path via ensure_src_on_path),
# then fall back to `src.ppe` (works if the repo root itself is on sys.path
# but src/ wasn't prepended, e.g. some IDE run configurations). If both fail,
# every name below is set to None and IMPORT_ERROR carries the reason, so
# `/health` and build_detector() can report a clear message instead of
# crashing the whole app at import time.
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
    """Read the ``PPE_DEVICE`` env var (``"cpu"``, ``"cuda"``, ``"0"``, ...), or None to auto-select."""
    env = os.environ.get("PPE_DEVICE")
    return env.strip() if env and env.strip() else None


def build_detector(
    weights: str | Path | None = None,
    conf: float = 0.25,
    device: str | None = None,
):
    """Construct a :class:`PPEDetector`, raising caller-friendly errors instead of crashing.

    Two failure modes are turned into specific, catchable exceptions so the
    API layer can map them to the right HTTP status (503, not 500):

    - ``RuntimeError`` if the ``ppe`` package itself failed to import
      (dependency/setup problem — see :data:`IMPORT_ERROR` for the cause).
    - ``FileNotFoundError`` if ``ppe`` imported fine but the resolved
      weights path doesn't exist on disk (data/config problem).

    ``device`` explicitly passed here wins over ``PPE_DEVICE``; leave it
    ``None`` to fall back to the environment variable via
    :func:`default_device`.
    """
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
    kwargs: dict = {"weights_path": str(path), "conf": float(conf)}
    chosen = device if device is not None else default_device()
    if chosen:
        kwargs["device"] = chosen
    return PPEDetector(**kwargs)
