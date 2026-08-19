"""Repo-root resolution and weight lookup for the local service."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_WEIGHT_CANDIDATES = (
    REPO_ROOT / "baselines" / "snehilsanyal_yolov8n_css" / "models" / "best.pt",
    REPO_ROOT / "models" / "best.pt",
)

CLIP_DIR = REPO_ROOT / "output" / "app"


def ensure_src_on_path() -> None:
    """Make ``ppe`` importable when the repo has not been pip-installed."""
    for path in (REPO_ROOT / "src", REPO_ROOT):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def resolve_weights_path(explicit: str | None = None) -> Path:
    """Explicit argument, then ``PPE_WEIGHTS``, then the checked-in candidates."""
    for candidate in (explicit, os.environ.get("PPE_WEIGHTS")):
        if candidate:
            path = Path(candidate).expanduser()
            return path if path.is_absolute() else (REPO_ROOT / path)
    for path in DEFAULT_WEIGHT_CANDIDATES:
        if path.is_file():
            return path
    return DEFAULT_WEIGHT_CANDIDATES[0]


ensure_src_on_path()
