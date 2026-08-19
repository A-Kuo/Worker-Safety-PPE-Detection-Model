"""Path, dataset, and import helpers shared by the pipeline scripts.

Label tables live in ``ppe.schema``. This module only resolves paths and reads
YOLO folder layouts.
"""

from __future__ import annotations

import csv
import importlib
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_DIR_NAMES = ("train", "valid", "val", "test")
SPLIT_CANON = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}

CONSTRUCTION_CLASS_NAMES = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
    "machinery",
    "vehicle",
]

# Cited Construction YOLOv8n epoch-99 numbers (Snehil Sanyal baseline).
KNOWN_EPOCH99 = {
    "precision": 0.900,
    "recall": 0.731,
    "map50": 0.809,
    "map50_95": 0.507,
}

NO_STAR_CLASSES = ("no_helmet", "no_vest", "no_goggles", "no_gloves", "no_mask")


def ensure_repo_on_path() -> None:
    """Make the src-layout ``ppe`` package importable from a plain checkout."""
    for path in (REPO_ROOT / "src", REPO_ROOT):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _import_ppe(module: str):
    ensure_repo_on_path()
    try:
        return importlib.import_module(f"ppe.{module}")
    except ImportError as exc:
        raise ImportError(
            f"Could not import ppe.{module}. Run `pip install -e .` from the repo root."
        ) from exc


def load_schema():
    return _import_ppe("schema")


def load_compliance():
    return _import_ppe("compliance")


def load_inference():
    return _import_ppe("inference")


def first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def resolve_baseline_weights() -> Path | None:
    return first_existing(
        REPO_ROOT / "baselines" / "snehilsanyal_yolov8n_css" / "models" / "best.pt",
        REPO_ROOT / "models" / "best.pt",
    )


def resolve_results_csv() -> Path | None:
    return first_existing(
        REPO_ROOT / "baselines" / "snehilsanyal_yolov8n_css" / "results" / "results.csv",
        REPO_ROOT / "results" / "results.csv",
    )


def resolve_confusion_matrix() -> Path | None:
    return first_existing(
        REPO_ROOT / "baselines" / "snehilsanyal_yolov8n_css" / "results" / "confusion_matrix.png",
        REPO_ROOT / "results" / "confusion_matrix.png",
    )


def count_images(root: Path) -> int:
    if not root.exists():
        return 0
    if root.is_file():
        return 1 if root.suffix.lower() in IMAGE_EXTS else 0
    return sum(1 for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def find_construction_image_root() -> Path | None:
    """Return a directory that actually contains Construction images, if any."""
    candidates = [
        REPO_ROOT / "data" / "raw" / "construction",
        REPO_ROOT / "data" / "construction",
        REPO_ROOT / "train",
        REPO_ROOT / "valid",
        REPO_ROOT / "data" / "train",
        REPO_ROOT / "data" / "valid",
    ]
    for cand in candidates:
        if count_images(cand) > 0:
            return cand
    return None


def find_construction_yaml() -> Path | None:
    return first_existing(
        REPO_ROOT / "data" / "raw" / "construction" / "data.yaml",
        REPO_ROOT / "configs" / "data" / "construction.yaml",
        REPO_ROOT / "data" / "data.yaml",
    )


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required (pip install pyyaml)") from exc
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required (pip install pyyaml)") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, default_flow_style=False)


def parse_names(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [str(raw[k]) for k in sorted(raw, key=lambda x: int(x))]
    if isinstance(raw, list):
        return [str(x) for x in raw]
    raise TypeError(f"Unsupported names field: {type(raw)}")


def read_dataset_names(yaml_path: Path) -> list[str]:
    return parse_names(load_yaml(yaml_path).get("names"))


def normalize_split_name(name: str) -> str:
    return SPLIT_CANON.get(name, name)


def discover_split_dirs(dataset_root: Path) -> dict[str, Path]:
    """Find train/valid/test image directories under a YOLO export."""
    found: dict[str, Path] = {}
    search_roots = [dataset_root]
    nested = [p for p in dataset_root.iterdir() if p.is_dir()] if dataset_root.is_dir() else []
    # Roboflow sometimes nests one extra project-version folder.
    search_roots.extend(nested)

    for root in search_roots:
        for raw in SPLIT_DIR_NAMES:
            for images_dir in (root / raw / "images", root / raw):
                if images_dir.is_dir() and count_images(images_dir) > 0:
                    found.setdefault(normalize_split_name(raw), images_dir)
                    break
    return found


def label_path_for_image(image_path: Path) -> Path:
    """Ultralytics convention: .../images/foo.jpg -> .../labels/foo.txt."""
    parts = list(image_path.parts)
    try:
        idx = max(i for i, part in enumerate(parts) if part.lower() == "images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    except ValueError:
        return image_path.with_suffix(".txt")


def iter_split_images(images_dir: Path) -> Iterator[Path]:
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def parse_yolo_label(label_path: Path) -> list[list[float]]:
    if not label_path.exists():
        return []
    rows: list[list[float]] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            continue
    return rows


def yolo_norm_to_xyxy(
    xc: float, yc: float, w: float, h: float
) -> tuple[float, float, float, float]:
    return xc - w / 2.0, yc - h / 2.0, xc + w / 2.0, yc + h / 2.0


def box_iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def parse_results_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            clean = {(k or "").strip(): (v or "").strip() for k, v in raw.items()}
            parsed: dict[str, float] = {}
            for key, value in clean.items():
                if not key or value == "":
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    continue
            if parsed:
                rows.append(parsed)
    return rows


def lookup_mapping(mapping: Mapping[str, str], raw_name: str) -> str | None:
    if raw_name in mapping:
        return mapping[raw_name]
    folded = {k.casefold(): v for k, v in mapping.items()}
    return folded.get(raw_name.casefold())


def mapping_for_source(schema, kind: str) -> dict[str, str]:
    attr = {
        "combined": "COMBINED_RAW_TO_UNIFIED",
        "construction": "CONSTRUCTION_TO_UNIFIED",
        "hhu": "HHU_TO_UNIFIED",
        "hardhat": "HHU_TO_UNIFIED",
    }[kind]
    value = getattr(schema, attr)
    return dict(value)


def infer_mapping_kind(names: Sequence[str]) -> str:
    nset = {n.casefold() for n in names}
    if "machinery" in nset or "vehicle" in nset:
        return "construction"
    if "head" in nset or "hi-viz vest" in nset or "hi-viz helmet" in nset:
        return "hhu"
    return "combined"


def build_id_map(
    schema, source_names: Sequence[str], name_mapping: Mapping[str, str]
) -> dict[int, int]:
    id_map: dict[int, int] = {}
    for old_id, raw in enumerate(source_names):
        unified = lookup_mapping(name_mapping, raw)
        if unified is None:
            continue
        id_map[old_id] = int(schema.unified_id(unified))
    return id_map


def apply_remap_line(
    schema,
    line: str,
    source_names: Sequence[str],
    name_mapping: Mapping[str, str],
) -> str | None:
    """Rewrite one YOLO label line, tolerating case differences in class names."""
    mapping = dict(name_mapping)
    for key, value in name_mapping.items():
        mapping.setdefault(key.casefold(), value)
    return schema.remap_yolo_line(line, list(source_names), mapping)


def write_unified_yaml(
    schema,
    dest: Path,
    *,
    train: str,
    val: str,
    test: str | None = None,
    names: Sequence[str] | None = None,
) -> None:
    """Write an Ultralytics dataset yaml on the unified class list.

    Split paths stay relative; Ultralytics resolves them against the yaml.
    """
    ordered = list(names or schema.UNIFIED_CLASS_NAMES)
    schema.write_dataset_yaml(dest, train, val, test or "", ordered)


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    try:
        dst.symlink_to(src.resolve())
        return "symlink"
    except OSError:
        pass
    try:
        dst.hardlink_to(src)
        return "hardlink"
    except OSError:
        import shutil

        shutil.copy2(src, dst)
        return "copy"


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"
