"""Unified 14-class PPE label schema and YOLO remapping helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

UNIFIED_CLASS_NAMES: list[str] = [
    "helmet",
    "no_helmet",
    "vest",
    "no_vest",
    "goggles",
    "no_goggles",
    "gloves",
    "no_gloves",
    "mask",
    "no_mask",
    "person",
    "cone",
    "ladder",
    "fall_detected",
]

COMBINED_RAW_TO_UNIFIED: dict[str, str] = {
    "Hardhat": "helmet",
    "NO-Hardhat": "no_helmet",
    "Safety Vest": "vest",
    "NO-Safety Vest": "no_vest",
    "Goggles": "goggles",
    "NO-Goggles": "no_goggles",
    "Gloves": "gloves",
    "NO-Gloves": "no_gloves",
    "Mask": "mask",
    "NO-Mask": "no_mask",
    "Person": "person",
    "Safety Cone": "cone",
    "Ladder": "ladder",
    "Fall-Detected": "fall_detected",
}

CONSTRUCTION_TO_UNIFIED: dict[str, str] = {
    "Hardhat": "helmet",
    "Mask": "mask",
    "NO-Hardhat": "no_helmet",
    "NO-Mask": "no_mask",
    "NO-Safety Vest": "no_vest",
    "Person": "person",
    "Safety Cone": "cone",
    "Safety Vest": "vest",
}

HHU_TO_UNIFIED: dict[str, str] = {
    "helmet": "helmet",
    "hi-viz helmet": "helmet",
    "hi-viz vest": "vest",
    "person": "person",
    "head": "no_helmet",
}

# Gap-fill fine-tuning data for vest/no_vest reliability: novest/no-vest-detect
# (Roboflow Universe), version 1. Exact class strings from its data.yaml —
# includes duplicate concepts under different spellings/casing (the dataset
# merges multiple annotation passes). "worker" has no dedicated unified class;
# treated as "person" (bounding boxes represent a person, same as elsewhere
# in this schema). lookup_mapping() case-folds, so "Vest"/"vest" share one entry.
GAP_VEST_TO_UNIFIED: dict[str, str] = {
    "Helmet": "helmet",
    "No-Helmet": "no_helmet",
    "no helmet": "no_helmet",
    "Vest": "vest",
    "No-Vest": "no_vest",
    "no vest": "no_vest",
    "Person": "person",
    "worker": "person",
}

# Classes present in both Construction v28 (after remap) and Combined PPE.
SHARED_EVAL_CLASSES: list[str] = [
    "helmet",
    "no_helmet",
    "vest",
    "no_vest",
    "mask",
    "no_mask",
    "person",
    "cone",
]

_UNIFIED_INDEX: dict[str, int] = {name: i for i, name in enumerate(UNIFIED_CLASS_NAMES)}


def unified_id(name: str) -> int:
    """Return the unified class index for ``name``."""
    try:
        return _UNIFIED_INDEX[name]
    except KeyError as exc:
        raise KeyError(f"Unknown unified class: {name!r}") from exc


def remap_yolo_line(line: str, raw_names: list[str], mapping: dict[str, str]) -> str | None:
    """Rewrite one YOLO label line to a unified class id.

    Return ``None`` if the line is empty or the class is dropped (unmapped).
    """
    stripped = line.strip()
    if not stripped:
        return None
    parts = stripped.split()
    try:
        raw_id = int(float(parts[0]))
    except (TypeError, ValueError):
        return None
    if raw_id < 0 or raw_id >= len(raw_names):
        return None
    unified_name = mapping.get(raw_names[raw_id])
    if unified_name is None:
        return None
    new_id = unified_id(unified_name)
    return " ".join([str(new_id), *parts[1:]])


def write_dataset_yaml(path, train, val, test, names) -> None:
    """Write an Ultralytics dataset YAML with train/val/test splits and class names."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = _coerce_names(names)
    lines = [
        f"train: {train}",
        f"val: {val}",
        f"test: {test}",
        f"nc: {len(ordered)}",
        "names:",
        *(_yaml_name_entries(ordered)),
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def _coerce_names(names: Sequence[str] | Mapping[int, str]) -> list[str]:
    if isinstance(names, Mapping):
        return [str(names[i]) for i in sorted(names)]
    return [str(n) for n in names]


def _yaml_name_entries(names: Sequence[str]) -> list[str]:
    entries: list[str] = []
    for name in names:
        escaped = name.replace("\\", "\\\\").replace('"', '\\"')
        if any(ch in name for ch in (":", "#", "{", "}", "[", "]", ",", "&", "*", "!", "|")):
            entries.append(f'  - "{escaped}"')
        else:
            entries.append(f"  - {name}")
    return entries
