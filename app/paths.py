"""Repo-root resolution and default weight lookup for the local demo."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ONNX / INT8 first so the demo prefers the NPU edge path when artifacts exist.
DEFAULT_WEIGHT_CANDIDATES = (
    REPO_ROOT / "models" / "best.int8.onnx",
    REPO_ROOT / "models" / "best.onnx",
    REPO_ROOT / "baselines" / "snehilsanyal_yolov8n_css" / "models" / "best.int8.onnx",
    REPO_ROOT / "baselines" / "snehilsanyal_yolov8n_css" / "models" / "best.onnx",
    REPO_ROOT / "baselines" / "snehilsanyal_yolov8n_css" / "models" / "best.pt",
    REPO_ROOT / "models" / "best.pt",
)

CLIP_DIR = REPO_ROOT / "output" / "app"


def ensure_src_on_path() -> None:
    """Prefer ``ppe`` via ``src/`` on path; repo root keeps ``src.ppe`` as fallback."""
    root = str(REPO_ROOT)
    src = str(REPO_ROOT / "src")
    # Insert root first, then src, so src ends at index 0 (ppe primary).
    for path in (root, src):
        if path not in sys.path:
            sys.path.insert(0, path)


def resolve_weights_path(explicit: str | None = None, *, prefer_onnx: bool = True) -> Path:
    """Resolve weights: explicit arg, then ``PPE_WEIGHTS``, then ONNX-first defaults."""
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_absolute() else (REPO_ROOT / path)
    env = os.environ.get("PPE_WEIGHTS")
    if env:
        path = Path(env).expanduser()
        return path if path.is_absolute() else (REPO_ROOT / path)
    candidates = DEFAULT_WEIGHT_CANDIDATES
    if not prefer_onnx:
        # Flip: .pt before .onnx siblings when torch is forced.
        pt = [c for c in candidates if c.suffix.lower() in {".pt", ".pth"}]
        onnx = [c for c in candidates if c.suffix.lower() == ".onnx"]
        candidates = tuple(pt + onnx)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return DEFAULT_WEIGHT_CANDIDATES[-2]  # baseline .pt as conventional fallback


ensure_src_on_path()
