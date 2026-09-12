#!/usr/bin/env python3
"""Fine-tune a pretrained external checkpoint on a small gap dataset.

Reads ``configs/finetune/<exp>.yaml``. Unlike ``scripts/train.py`` (which
starts every experiment from Ultralytics COCO-pretrained ``yolov8n.pt`` /
``yolov8s.pt``), this script *requires* a local ``base_checkpoint`` — an
already-fetched, pinned external checkpoint (see ``scripts/fetch_checkpoint.py``)
— and never silently falls back to a COCO default, since that would corrupt
the raw-checkpoint-vs-fine-tuned-vs-from-scratch comparison this pipeline
exists to produce.

Before training, verifies the base checkpoint's class names match the
fine-tune dataset yaml's names 1:1 (count and order) — a name/order mismatch
means Ultralytics would silently retrain a misaligned detection head.

``--dry-run`` prints the resolved kwargs and exits without starting
``YOLO.train``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _common import (  # noqa: E402
    REPO_ROOT,
    first_existing,
    load_schema,
    load_yaml,
    mapped_unified_names,
    read_dataset_names,
)

CONFIG_DIR = REPO_ROOT / "configs" / "finetune"
# Keys that are documentation / our CLI, not Ultralytics train() kwargs.
META_KEYS = {"experiment", "base_checkpoint"}


def _available_experiments() -> list[str]:
    if not CONFIG_DIR.is_dir():
        return []
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True, help="configs/finetune/<exp>.yaml")
    parser.add_argument("--config", type=Path, default=None, help="Override yaml path")
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=None,
        help="Override base_checkpoint (local .pt path, never a repo_id)",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt in the run dir")
    parser.add_argument("--last", type=Path, default=None, help="Explicit checkpoint to resume")
    parser.add_argument("--data", default=None, help="Override data yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved train kwargs and exit without starting YOLO.train",
    )
    return parser.parse_args()


def resolve_config_path(exp: str, override: Path | None) -> Path:
    if override:
        path = override if override.is_absolute() else REPO_ROOT / override
        if not path.exists():
            raise SystemExit(f"Config not found: {path}")
        return path
    path = CONFIG_DIR / f"{exp}.yaml"
    if not path.exists():
        available = _available_experiments()
        hint = f"Available: {', '.join(available)}" if available else f"No configs found under {CONFIG_DIR}"
        raise SystemExit(f"Missing {path}. {hint}")
    return path


def resolve_base_checkpoint(cfg: dict[str, Any], override: Path | None) -> Path:
    raw = override or cfg.get("base_checkpoint")
    if not raw:
        raise SystemExit(
            "base_checkpoint is required (config key or --base-checkpoint). "
            "Fetch one first with scripts/fetch_checkpoint.py — this script never "
            "falls back to a COCO-pretrained default."
        )
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise SystemExit(
            f"base_checkpoint not found: {path}. "
            "Fetch it first with scripts/fetch_checkpoint.py."
        )
    return path


def load_train_kwargs(path: Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    if isinstance(cfg.get("finetune"), dict) and "data" not in cfg:
        cfg = dict(cfg["finetune"])
    for key in META_KEYS:
        cfg.pop(key, None)
    data = cfg.get("data")
    if isinstance(data, str) and not Path(data).is_absolute():
        resolved = REPO_ROOT / data
        if resolved.exists():
            cfg["data"] = str(resolved)
    return cfg


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.data:
        cfg["data"] = args.data
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch is not None:
        cfg["batch"] = args.batch
    if args.device is not None:
        cfg["device"] = args.device
    if args.project:
        cfg["project"] = args.project
    if args.name:
        cfg["name"] = args.name
    cfg.setdefault("project", str(REPO_ROOT / "runs" / "finetune"))
    cfg.setdefault("name", args.exp)
    cfg.setdefault("exist_ok", True)
    return cfg


def find_last_ckpt(cfg: dict[str, Any], explicit: Path | None) -> Path | None:
    if explicit:
        path = explicit if explicit.is_absolute() else REPO_ROOT / explicit
        return path if path.exists() else None
    project = Path(cfg.get("project", REPO_ROOT / "runs" / "finetune"))
    name = str(cfg.get("name", "exp"))
    return first_existing(project / name / "weights" / "last.pt", project / name / "last.pt")


def check_schema_compatibility(base_checkpoint: Path, data_yaml: Path) -> None:
    """Fail loudly on a class order mismatch instead of retraining a misaligned head.

    Compares *positions*, not literal strings: a raw third-party checkpoint's
    class names are first mapped to unified schema names (position N's
    concept), then checked against the fine-tune dataset yaml's `names:` list
    at the same positions. Ultralytics keeps pretrained detection-head
    weights as the fine-tune initialization when `nc` matches, regardless of
    whether the class *order* lines up — a silent mismatch there means
    position 0 meant "helmet" during pretraining but is being asked to mean
    "fall_detected" during fine-tuning, a bad initialization prior, not a
    crash. So: the fine-tune dataset yaml's `names:` order must match the
    base checkpoint's own (unified-mapped) class order — not necessarily
    UNIFIED_CLASS_NAMES's canonical order.
    """
    from ultralytics import YOLO

    schema = load_schema()
    base_names_raw = YOLO(str(base_checkpoint)).names
    base_names = [str(base_names_raw[i]) for i in sorted(base_names_raw)]
    try:
        mapped_names = mapped_unified_names(schema, base_names)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    data_names = read_dataset_names(data_yaml)

    if not data_names:
        raise SystemExit(f"Could not read class names from {data_yaml}")
    if len(mapped_names) != len(data_names):
        raise SystemExit(
            f"Class count mismatch: base_checkpoint has {len(mapped_names)} classes "
            f"(mapped: {mapped_names}), but {data_yaml} declares {len(data_names)} "
            f"classes {data_names}. Fine-tuning requires matching head sizes."
        )
    if mapped_names != data_names:
        raise SystemExit(
            "Class order mismatch between base_checkpoint and the fine-tune dataset yaml "
            "(compared after mapping the checkpoint's raw names to unified concepts):\n"
            f"  base_checkpoint (raw):     {base_names}\n"
            f"  base_checkpoint (mapped):  {mapped_names}\n"
            f"  {data_yaml.name}: {data_names}\n"
            "Same count but different order — Ultralytics would keep the pretrained head "
            "weights as a bad-order initialization rather than crash. Write the dataset "
            "yaml's `names:` order to match the base checkpoint's mapped order shown above "
            "(not necessarily UNIFIED_CLASS_NAMES's canonical order)."
        )
    print(f"Schema check OK: {len(mapped_names)} classes match position-for-position.")


def main() -> int:
    args = _parse_args()
    cfg_path = resolve_config_path(args.exp, args.config)
    cfg = load_train_kwargs(cfg_path)
    raw_cfg_for_checkpoint = load_yaml(cfg_path)
    base_checkpoint = resolve_base_checkpoint(raw_cfg_for_checkpoint, args.base_checkpoint)
    cfg = apply_overrides(cfg, args)

    data = cfg.get("data")
    if not data:
        raise SystemExit("Config must set `data:` (a dataset yaml path).")
    data_yaml = Path(data)
    if not data_yaml.is_absolute():
        data_yaml = REPO_ROOT / data_yaml
    if not data_yaml.is_file():
        raise SystemExit(f"data yaml not found: {data_yaml}")

    model_name = str(base_checkpoint)

    if args.resume:
        last = find_last_ckpt(cfg, args.last)
        if last is None:
            raise SystemExit("--resume set but last.pt was not found. Pass --last path/to/last.pt")
        model_name = str(last)
        cfg["resume"] = True
    else:
        check_schema_compatibility(base_checkpoint, data_yaml)

    print(f"Experiment:      {args.exp}")
    print(f"Config:          {cfg_path}")
    print(f"Base checkpoint: {model_name}")
    print("Train kwargs:")
    for key in sorted(cfg):
        print(f"  {key}: {cfg[key]}")

    if args.dry_run:
        print("Dry-run: not calling YOLO.train")
        return 0

    from ultralytics import YOLO

    model = YOLO(model_name)
    model.train(**cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
