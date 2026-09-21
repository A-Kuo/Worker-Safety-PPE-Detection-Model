"""Per-class confidence thresholds.

One global ``conf`` cannot suit every class. ``no_vest`` is the motivating case: the model finds
most ground-truth ``no_vest`` boxes but scores them low (about 70% are detected at conf >= 0.01,
only ~24% clear 0.25), so a 0.25 global floor throws most of them away. A per-class override
such as ``no_vest=0.10`` recovers recall for that class only, at a known false-positive cost,
without lowering the bar for classes that are already well calibrated.

Backends are asked for detections at ``effective_floor`` (the lowest threshold in play) and the
results are then filtered per class with ``filter_detections``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from ppe.compliance import Detection
from ppe.schema import UNIFIED_CLASS_NAMES


def parse_class_conf(text: str | None) -> dict[str, float]:
    """Parse ``"no_vest=0.05,no_helmet=0.1"``; empty/None gives ``{}``. Raises ValueError on bad input."""
    result: dict[str, float] = {}
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        name, sep, raw = part.partition("=")
        name, raw = name.strip(), raw.strip()
        if not sep or not raw:
            raise ValueError(f"class threshold {part!r} must look like name=0.10")
        if name not in UNIFIED_CLASS_NAMES:
            raise ValueError(f"unknown class {name!r} in class thresholds; choose from {list(UNIFIED_CLASS_NAMES)}")
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"threshold for {name!r} is not a number: {raw!r}") from exc
        if not 0.0 < value <= 1.0:
            raise ValueError(f"threshold for {name!r} must be in (0, 1], got {value}")
        result[name] = value
    return result


def effective_floor(default: float, class_conf: Mapping[str, float] | None) -> float:
    """Lowest confidence any class needs; what the backend should be asked for."""
    return min([default, *(class_conf or {}).values()])


def filter_detections(
    detections: Iterable[Detection], default: float, class_conf: Mapping[str, float] | None
) -> list[Detection]:
    """Keep a detection if its confidence meets its own class's threshold (else ``default``)."""
    overrides = class_conf or {}
    return [d for d in detections if d.conf >= overrides.get(d.cls_name, default)]
