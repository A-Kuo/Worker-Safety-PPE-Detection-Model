"""Unified 14-class PPE label schema and YOLO remapping helpers.

Three source datasets ship with their own, incompatible label vocabularies
(different names, different orderings, different class counts). Everything
in this module exists to translate those raw vocabularies onto one
**unified 14-class schema** so a single detector can be trained and scored
consistently, and so downstream code (compliance, inference, the app) only
has to know about one set of names.

Class order matters: ``UNIFIED_CLASS_NAMES[i]`` is class id ``i`` for every
unified-schema label file and every exported model. Don't reorder it after
data has been remapped or trained on, or ids will silently point at the
wrong class.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

# The single source of truth for class ids. Position in this list *is* the
# YOLO class id (helmet=0, no_helmet=1, ...). Positive/negative PPE pairs are
# kept adjacent (helmet, no_helmet) so "did they have it or not" is a +1 away.
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

# Roboflow "Personal Protective Equipment Combined Model v4" — our primary
# 44k-image training set. All 14 unified classes exist natively here, so
# this mapping is effectively a rename/normalize step (raw name -> snake_case).
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

# Roboflow "Construction Site Safety v28" — the inherited third-party
# baseline (10 raw classes). `machinery` and `vehicle` have no unified
# equivalent and are intentionally left out of this mapping, so
# `remap_yolo_line` drops any box labeled with them (see docstring below).
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

# Hard Hat Universe — held-out cross-domain evaluation set only (never
# trained on). `head` has no explicit "missing helmet" label in the source
# data, so we treat a bare head detection as an implicit `no_helmet`. This is
# an assumption, not ground truth — score it separately (see
# docs/data_distribution.md) so it doesn't quietly bias Combined metrics.
HHU_TO_UNIFIED: dict[str, str] = {
    "helmet": "helmet",
    "hi-viz helmet": "helmet",
    "hi-viz vest": "vest",
    "person": "person",
    "head": "no_helmet",
}

# The subset of unified classes present on *both* Construction v28 (after
# remap) and Combined PPE. Use this list — not the full 14 classes — when
# comparing a Combined-trained model against the inherited Construction
# baseline, since Construction never had goggles/gloves/ladder/fall labels.
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

# Precomputed name -> index lookup so `unified_id` is O(1) instead of
# scanning UNIFIED_CLASS_NAMES on every call (this runs per label line,
# across tens of thousands of images during remap).
_UNIFIED_INDEX: dict[str, int] = {name: i for i, name in enumerate(UNIFIED_CLASS_NAMES)}


def unified_id(name: str) -> int:
    """Return the unified class index for ``name``.

    Raises ``KeyError`` (with a clearer message than a bare dict lookup)
    if ``name`` isn't one of the 14 :data:`UNIFIED_CLASS_NAMES` — this is
    almost always a typo or a raw (non-remapped) class name being passed in
    by mistake.
    """
    try:
        return _UNIFIED_INDEX[name]
    except KeyError as exc:
        raise KeyError(f"Unknown unified class: {name!r}") from exc


def remap_yolo_line(line: str, raw_names: list[str], mapping: dict[str, str]) -> str | None:
    """Rewrite one YOLO label line to a unified class id.

    A YOLO label line looks like ``"<class_id> <x_center> <y_center> <width>
    <height>"`` (all normalized 0-1 except the id). This swaps the leading
    raw ``class_id`` for its unified equivalent and leaves the box geometry
    untouched — the whole point is that pixel coordinates don't change, only
    which vocabulary the class number points into.

    Returns ``None`` (meaning: drop this line) when:
    - the line is empty/whitespace-only,
    - the leading token isn't a valid integer,
    - the raw id is out of range for ``raw_names``, or
    - the raw class name has no entry in ``mapping`` (e.g. Construction's
      ``machinery`` / ``vehicle``, which don't exist in the unified schema).

    Callers should skip ``None`` results rather than write them out — a
    dropped detection is not the same as an empty label file.
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
    """Write an Ultralytics dataset YAML with train/val/test splits and class names.

    ``train``/``val``/``test`` are plain strings (paths or URLs) written
    verbatim — this function does not validate that they exist on disk.
    ``names`` may be a list (position = class id) or a ``{id: name}`` dict;
    either way the YAML's ``names:`` block is written in ascending id order,
    which is what Ultralytics expects.
    """
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
    """Normalize either a list or an ``{id: name}`` mapping into an ordered list."""
    if isinstance(names, Mapping):
        return [str(names[i]) for i in sorted(names)]
    return [str(n) for n in names]


def _yaml_name_entries(names: Sequence[str]) -> list[str]:
    """Format each class name as a YAML list item, quoting only when needed.

    Most class names (``helmet``, ``no_vest``, ...) are safe as bare YAML
    scalars. Anything containing a YAML-significant character gets
    double-quoted with minimal escaping so the file stays valid without
    quoting every single line.
    """
    entries: list[str] = []
    for name in names:
        escaped = name.replace("\\", "\\\\").replace('"', '\\"')
        if any(ch in name for ch in (":", "#", "{", "}", "[", "]", ",", "&", "*", "!", "|")):
            entries.append(f'  - "{escaped}"')
        else:
            entries.append(f"  - {name}")
    return entries
