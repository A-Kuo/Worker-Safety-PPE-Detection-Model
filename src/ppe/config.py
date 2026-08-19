"""Runtime settings for the edge detector.

Everything the pipeline needs to run on a device is collected here so a
deployment can be described by one object (or one set of environment
variables) instead of scattered keyword arguments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from ppe.schema import UNIFIED_CLASS_NAMES

ENV_PREFIX = "PPE_"

DEFAULT_REQUIRED_PPE = ("helmet", "vest")
DEFAULT_IMAGE_SIZE = 640


@dataclass(frozen=True)
class RuntimeConfig:
    """Detector, tracker, and alerting knobs for one deployment."""

    weights: Path | None = None
    backend: str = "auto"
    device: str | None = None
    imgsz: int = DEFAULT_IMAGE_SIZE
    conf: float = 0.25
    iou: float = 0.45
    max_detections: int = 300

    required_ppe: tuple[str, ...] = DEFAULT_REQUIRED_PPE
    associate_iou: float = 0.05

    track_iou: float = 0.3
    track_max_age: int = 15

    alert_min_frames: int = 3
    alert_cooldown_s: float = 10.0

    frame_stride: int = 1
    warmup_frames: int = 1

    extra: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.conf < 1.0:
            raise ValueError(f"conf must be in (0, 1), got {self.conf}")
        if not 0.0 < self.iou < 1.0:
            raise ValueError(f"iou must be in (0, 1), got {self.iou}")
        if self.imgsz <= 0 or self.imgsz % 32 != 0:
            raise ValueError(f"imgsz must be a positive multiple of 32, got {self.imgsz}")
        if self.frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {self.frame_stride}")
        if self.alert_min_frames < 1:
            raise ValueError(f"alert_min_frames must be >= 1, got {self.alert_min_frames}")
        unknown = [name for name in self.required_ppe if name not in UNIFIED_CLASS_NAMES]
        if unknown:
            raise ValueError(f"required_ppe contains unknown classes: {unknown}")

    @property
    def backend_name(self) -> str:
        """Concrete backend, resolving ``auto`` from the weights extension."""
        if self.backend != "auto":
            return self.backend
        suffix = self.weights.suffix.lower() if self.weights else ""
        if suffix == ".onnx":
            return "onnx"
        return "ultralytics"

    def with_overrides(self, **kwargs) -> RuntimeConfig:
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **clean)


def config_from_env(env: dict[str, str] | None = None) -> RuntimeConfig:
    """Build a config from ``PPE_*`` variables, falling back to the defaults."""
    source = dict(os.environ if env is None else env)

    def read(name: str) -> str | None:
        value = source.get(ENV_PREFIX + name)
        return value.strip() if value and value.strip() else None

    weights = read("WEIGHTS")
    required = read("REQUIRED_PPE")

    return RuntimeConfig(
        weights=Path(weights).expanduser() if weights else None,
        backend=read("BACKEND") or "auto",
        device=read("DEVICE"),
        imgsz=int(read("IMGSZ") or DEFAULT_IMAGE_SIZE),
        conf=float(read("CONF") or 0.25),
        iou=float(read("IOU") or 0.45),
        required_ppe=_parse_required(required),
        track_iou=float(read("TRACK_IOU") or 0.3),
        track_max_age=int(read("TRACK_MAX_AGE") or 15),
        alert_min_frames=int(read("ALERT_MIN_FRAMES") or 3),
        alert_cooldown_s=float(read("ALERT_COOLDOWN") or 10.0),
        frame_stride=int(read("FRAME_STRIDE") or 1),
    )


def _parse_required(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return DEFAULT_REQUIRED_PPE
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    return names or DEFAULT_REQUIRED_PPE
