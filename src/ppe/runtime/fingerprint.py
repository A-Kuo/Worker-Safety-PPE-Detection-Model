"""Runtime fingerprint: everything needed to reproduce (or distrust) a latency number.

Every field is best-effort and never raises: a missing optional package yields ``None`` rather than
breaking inference. Nothing here imports torch.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

# Distributions that provide the ``onnxruntime`` import; only one is normally installed.
_ORT_DISTS = (
    "onnxruntime-openvino",
    "onnxruntime-qnn",
    "onnxruntime-directml",
    "onnxruntime-gpu",
    "onnxruntime",
)


def file_sha256(path: str | Path, *, chunk: int = 1 << 20) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while block := fh.read(chunk):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def ort_distribution() -> dict[str, str | None]:
    """Which ORT build is installed (distribution name + version), e.g. onnxruntime-openvino 1.20."""
    for dist in _ORT_DISTS:
        version = _dist_version(dist)
        if version:
            return {"distribution": dist, "version": version}
    return {"distribution": None, "version": None}


def onnx_opset(path: str | Path) -> int | None:
    """Default-domain opset of an ONNX file, or None when ``onnx`` isn't installed / unreadable."""
    try:
        import onnx

        model = onnx.load(str(path), load_external_data=False)
        for imp in model.opset_import:
            if imp.domain in ("", "ai.onnx"):
                return int(imp.version)
    except Exception:  # noqa: BLE001 - optional enrichment only
        return None
    return None


@lru_cache(maxsize=1)
def openvino_devices() -> dict[str, Any] | None:
    """OpenVINO runtime version and per-device names (e.g. the NPU's marketing name), if installed."""
    try:
        import openvino as ov

        core = ov.Core()
        devices: dict[str, Any] = {}
        for dev in core.available_devices:
            entry: dict[str, Any] = {}
            try:
                entry["name"] = str(core.get_property(dev, "FULL_DEVICE_NAME"))
            except Exception:  # noqa: BLE001
                pass
            if dev.startswith("NPU"):
                try:
                    entry["driver_version"] = str(core.get_property(dev, "NPU_DRIVER_VERSION"))
                except Exception:  # noqa: BLE001
                    pass
            devices[dev] = entry
        return {"version": getattr(ov, "__version__", None), "devices": devices}
    except Exception:  # noqa: BLE001
        return None


def runtime_fingerprint(model_path: str | Path | None = None) -> dict[str, Any]:
    """Host + runtime + (optionally) model identity."""
    try:
        import onnxruntime as ort

        ort_info = {**ort_distribution(), "module_version": ort.__version__,
                    "available_providers": list(ort.get_available_providers())}
    except ImportError:
        ort_info = {"distribution": None, "version": None, "module_version": None,
                    "available_providers": []}
    fp: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "onnxruntime": ort_info,
        "openvino": openvino_devices(),
    }
    if model_path is not None:
        p = Path(model_path)
        if p.is_file():
            st = p.stat()
            sha, opset = _model_identity(str(p), st.st_mtime_ns, st.st_size)
            fp["model"] = {"path": str(p), "bytes": st.st_size, "sha256": sha, "opset": opset}
        else:
            fp["model"] = {"path": str(p), "bytes": None, "sha256": None, "opset": None}
    return fp


@lru_cache(maxsize=32)
def _model_identity(path: str, mtime_ns: int, size: int) -> tuple[str | None, int | None]:
    """SHA-256 and opset of a model file, cached by (path, mtime, size).

    Hashing re-reads the whole file and ``onnx.load`` parses the full protobuf, so doing it for the main
    model and every specialist at each session open was avoidable startup cost. mtime/size are part of
    the key, so an edited or replaced file is re-hashed.
    """
    return file_sha256(path), onnx_opset(path)


def scrub_paths(obj: Any, root: str | Path) -> Any:
    """Replace absolute path prefixes in every string: repo ``root`` -> ``<repo>``, home -> ``~``.

    Result files get committed as evidence; absolute paths leak the author's directory layout and make
    otherwise identical runs differ between machines. The home directory is scrubbed too, so a model
    loaded from outside the project (e.g. ``~/.ppe/models``) does not leak the user name.
    """
    pairs = [(str(root), "<repo>"), (str(Path.home()), "~")]
    return _scrub(obj, [(p.replace("\\", "/"), label) for p, label in pairs if p and p not in ("/", ".")])


def _scrub(obj: Any, pairs: list[tuple[str, str]]) -> Any:
    if isinstance(obj, dict):
        return {k: _scrub(v, pairs) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub(v, pairs) for v in obj]
    if isinstance(obj, str):
        norm = obj.replace("\\", "/")
        for prefix, label in pairs:
            if norm == prefix or norm.startswith(prefix + "/"):
                return label + norm[len(prefix):]
    return obj
