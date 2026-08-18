"""Repo-root resolution and default weight lookup for the local demo.

The API and UI both need to find two things regardless of the working
directory they're launched from: the ``ppe`` package (which lives under
``src/``, not installed as a wheel by default) and a model weights file.
This module centralizes both lookups so `app/api/main.py` and
`app/ui/streamlit_app.py` don't duplicate path logic.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Two levels up from this file: app/paths.py -> app/ -> repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Checked in order when no explicit weights path or PPE_WEIGHTS is given.
# The first is the inherited Construction baseline (ships with the repo);
# the second is a conventional local drop-in spot for a custom checkpoint.
DEFAULT_WEIGHT_CANDIDATES = (
    REPO_ROOT / "baselines" / "snehilsanyal_yolov8n_css" / "models" / "best.pt",
    REPO_ROOT / "models" / "best.pt",
)

# Where /predict/video?return_clip=true writes annotated MP4s for later download.
CLIP_DIR = REPO_ROOT / "output" / "app"


def ensure_src_on_path() -> None:
    """Make ``import ppe`` work regardless of the current working directory or install state.

    Prefers ``ppe`` (via ``src/`` on ``sys.path``) over the ``src.ppe``
    fallback import path used elsewhere — inserting ``root`` first and
    ``src`` second means ``src`` ends up at index 0, so `import ppe`
    resolves before `import src.ppe` would even be tried. Idempotent: safe
    to call multiple times (e.g. once at module import, once defensively in
    each entry point) without duplicating `sys.path` entries.
    """
    root = str(REPO_ROOT)
    src = str(REPO_ROOT / "src")
    # Insert root first, then src, so src ends at index 0 (ppe primary).
    for path in (root, src):
        if path not in sys.path:
            sys.path.insert(0, path)


def resolve_weights_path(explicit: str | None = None) -> Path:
    """Resolve a weights path with a clear precedence order.

    1. ``explicit`` argument (e.g. a `?weights=` query param or CLI flag),
    2. the ``PPE_WEIGHTS`` environment variable,
    3. the first existing file in :data:`DEFAULT_WEIGHT_CANDIDATES`,
    4. otherwise, the *first* default candidate (even though it doesn't
       exist) — so callers get a predictable, loggable path to report as
       "missing" rather than ``None``.

    Relative paths (explicit or via env var) are resolved against
    :data:`REPO_ROOT`, not the current working directory, so the demo
    behaves the same whether it's launched from the repo root or elsewhere.
    """
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_absolute() else (REPO_ROOT / path)
    env = os.environ.get("PPE_WEIGHTS")
    if env:
        path = Path(env).expanduser()
        return path if path.is_absolute() else (REPO_ROOT / path)
    for candidate in DEFAULT_WEIGHT_CANDIDATES:
        if candidate.is_file():
            return candidate
    return DEFAULT_WEIGHT_CANDIDATES[0]


# Run once at import time so anything importing app.paths can immediately
# `import ppe` without remembering to call this itself.
ensure_src_on_path()
