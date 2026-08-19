"""One shared pipeline for the API and the Streamlit UI.

Loading a checkpoint costs seconds and hundreds of megabytes, so the process
keeps a single pipeline and rebuilds it only when the weights change. The lock
is exported because inference itself is not reentrant.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

from app.paths import ensure_src_on_path, resolve_weights_path

ensure_src_on_path()

from ppe.config import RuntimeConfig, config_from_env  # noqa: E402
from ppe.pipeline import EdgePipeline  # noqa: E402

PIPELINE_LOCK = threading.Lock()

_pipeline: EdgePipeline | None = None
_pipeline_key: tuple[str, str] | None = None
_injected = False


@dataclass(frozen=True)
class RuntimeStatus:
    """What ``/health`` needs to explain why the service is or is not ready."""

    ready: bool
    weights: Path
    weights_exist: bool
    backend: str
    device: str | None
    detail: str | None = None


def default_device() -> str | None:
    value = os.environ.get("PPE_DEVICE", "").strip()
    return value or None


def build_config(weights: str | Path | None = None, conf: float | None = None) -> RuntimeConfig:
    return config_from_env().with_overrides(
        weights=resolve_weights_path(str(weights) if weights else None),
        conf=conf,
        device=default_device(),
    )


def get_pipeline(weights: str | Path | None = None, conf: float | None = None) -> EdgePipeline:
    """Return the shared pipeline, reloading only when the weights path changes."""
    global _pipeline, _pipeline_key

    if _injected and _pipeline is not None:
        return _apply_conf(_pipeline, conf)

    config = build_config(weights, conf)
    if config.weights is None or not config.weights.is_file():
        raise FileNotFoundError(
            f"Weights not found at {config.weights}. Set PPE_WEIGHTS or pass a weights path."
        )

    key = (str(config.weights), config.backend_name)
    with PIPELINE_LOCK:
        if _pipeline is None or _pipeline_key != key:
            _pipeline = EdgePipeline.from_config(config)
            _pipeline_key = key
            return _pipeline
    return _apply_conf(_pipeline, config.conf)


def set_pipeline(pipeline: EdgePipeline) -> None:
    """Install a pipeline directly, bypassing weight resolution.

    Tests use this to serve a scripted backend, and it is also how an offline
    demo runs without a checkpoint on disk.
    """
    global _pipeline, _pipeline_key, _injected
    with PIPELINE_LOCK:
        _pipeline = pipeline
        _pipeline_key = ("injected", "injected")
        _injected = True


def clear_pipeline() -> None:
    global _pipeline, _pipeline_key, _injected
    with PIPELINE_LOCK:
        _pipeline = None
        _pipeline_key = None
        _injected = False


def runtime_status() -> RuntimeStatus:
    weights = resolve_weights_path()
    exists = weights.is_file()
    loaded = _pipeline is not None
    detail = None if exists or loaded else f"Weights missing at {weights}. Set PPE_WEIGHTS."
    return RuntimeStatus(
        ready=loaded or exists,
        weights=weights,
        weights_exist=exists,
        backend=build_config().backend_name,
        device=default_device(),
        detail=detail,
    )


def _apply_conf(pipeline: EdgePipeline, conf: float | None) -> EdgePipeline:
    if conf is None:
        return pipeline
    with PIPELINE_LOCK:
        pipeline.config = pipeline.config.with_overrides(conf=float(conf))
        if hasattr(pipeline.backend, "conf"):
            pipeline.backend.conf = float(conf)
    return pipeline
