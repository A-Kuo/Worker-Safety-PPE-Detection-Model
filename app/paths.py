"""Repo-root resolution and weight lookup for the local service."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Inference runs on ONNX Runtime, so the exported models come first. The
# inherited .pt is last and only resolves so /health can say why it is wrong.
DEFAULT_WEIGHT_CANDIDATES = (
    REPO_ROOT / "models" / "best.int8.onnx",
    REPO_ROOT / "models" / "best.onnx",
    REPO_ROOT / "baselines" / "snehilsanyal_yolov8n_css" / "models" / "best.pt",
)

CLIP_DIR = REPO_ROOT / "output" / "app"


def ensure_src_on_path() -> None:
    """Make ``ppe`` importable when the repo has not been pip-installed."""
    for path in (REPO_ROOT / "src", REPO_ROOT):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def resolve_weights_path(explicit: str | None = None) -> Path:
    """Explicit argument, then ``PPE_WEIGHTS``, then the exported models."""
    for candidate in (explicit, os.environ.get("PPE_WEIGHTS")):
        if candidate:
            path = Path(candidate).expanduser()
            return path if path.is_absolute() else (REPO_ROOT / path)
    for path in DEFAULT_WEIGHT_CANDIDATES:
        if path.is_file():
            return path
    return DEFAULT_WEIGHT_CANDIDATES[0]


ensure_src_on_path()
