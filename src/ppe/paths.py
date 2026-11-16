"""Where the package looks for the project and for models. Works from a checkout *and* a wheel.

The package used to find "the repo" by counting parent directories from ``__file__``. After
``pip install`` that points into ``site-packages``, so default model lookup and every relative path
broke. Resolution order is now explicit:

- ``project_root()``: ``PPE_HOME`` if set, else the source checkout this file lives in (editable or
  in-tree use), else the current working directory (wheel install).
- ``model_dir()``: ``PPE_MODEL_DIR`` if set, else ``~/.ppe/models`` (where a fetched model is stored).
"""

from __future__ import annotations

import os
from pathlib import Path


def source_root() -> Path | None:
    """Root of the source checkout containing this file, or None when installed from a wheel."""
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "ppe").is_dir():
        return candidate
    return None


def project_root() -> Path:
    """Directory that relative model/source/output paths are resolved against."""
    env = os.environ.get("PPE_HOME", "").strip()
    if env:
        return Path(env).expanduser()
    return source_root() or Path.cwd()


def model_dir() -> Path:
    """Per-user model store (not created here)."""
    env = os.environ.get("PPE_MODEL_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".ppe" / "models"
