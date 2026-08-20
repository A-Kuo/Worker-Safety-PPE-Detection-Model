#!/usr/bin/env python3
"""CLI wrapper around Ultralytics ``YOLO.train`` for the E0–E4 configs.

Reads ``configs/train/{e0_n,e1_s,e2_focal,e3_augs,e4_full44k}.yaml``.
Resume-friendly. ``--dry-run`` prints the resolved kwargs and exits - this
script does not start a run unless you omit ``--dry-run``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _common import REPO_ROOT, first_existing, load_yaml  # noqa: E402

EXPERIMENTS = ("e0_n", "e1_s", "e2_focal", "e3_augs", "e4_full44k")
CONFIG_DIR = REPO_ROOT / "configs" / "train"
# Keys that are documentation / our CLI, not Ultralytics train() kwargs.
META_KEYS = {"experiment", "subset_size"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", choices=EXPERIMENTS, required=True)
    parser.add_argument("--config", type=Path, default=None, help="Override yaml path")
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt in the run dir")
    parser.add_argument("--last", type=Path, default=None, help="Explicit checkpoint to resume")
    parser.add_argument("--model", default=None, help="Override model weights / yaml")
    parser.add_argument("--data", default=None, help="Override data yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Override batch. Use 8 on 8GB local GPUs; 16–32 on Colab/Kaggle P100.",
    )
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
        raise SystemExit(
            f"Missing {path}. Expected experiment yamls: "
            + ", ".join(f"configs/train/{e}.yaml" for e in EXPERIMENTS)
        )
    return path


def load_train_kwargs(path: Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    # Allow a nested `train:` block or a flat Ultralytics kwargs file.
    if isinstance(cfg.get("train"), dict) and "data" not in cfg:
        cfg = dict(cfg["train"])
    for key in META_KEYS:
        cfg.pop(key, None)
    data = cfg.get("data")
    if isinstance(data, str) and not Path(data).is_absolute():
        resolved = REPO_ROOT / data
        if resolved.exists():
            cfg["data"] = str(resolved)
    return cfg


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    model = args.model or cfg.pop("model", None) or _default_model(args.exp)
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
    cfg.setdefault("project", str(REPO_ROOT / "runs" / "train"))
    cfg.setdefault("name", args.exp)
    cfg.setdefault("exist_ok", True)
    return str(model), cfg


def _default_model(exp: str) -> str:
    if exp == "e1_s":
        return "yolov8s.pt"
    return "yolov8n.pt"


def find_last_ckpt(cfg: dict[str, Any], explicit: Path | None) -> Path | None:
    if explicit:
        path = explicit if explicit.is_absolute() else REPO_ROOT / explicit
        return path if path.exists() else None
    project = Path(cfg.get("project", REPO_ROOT / "runs" / "train"))
    name = str(cfg.get("name", "exp"))
    return first_existing(project / name / "weights" / "last.pt", project / name / "last.pt")


def main() -> int:
    args = _parse_args()
    cfg_path = resolve_config_path(args.exp, args.config)
    cfg = load_train_kwargs(cfg_path)
    model_name, train_kwargs = apply_overrides(cfg, args)

    if args.resume:
        last = find_last_ckpt(train_kwargs, args.last)
        if last is None:
            raise SystemExit(" --resume set but last.pt was not found. Pass --last path/to/last.pt")
        model_name = str(last)
        train_kwargs["resume"] = True

    print(f"Experiment: {args.exp}")
    print(f"Config:     {cfg_path}")
    print(f"Model:      {model_name}")
    print("Train kwargs:")
    for key in sorted(train_kwargs):
        print(f"  {key}: {train_kwargs[key]}")

    if args.dry_run:
        print("Dry-run: not calling YOLO.train")
        return 0

    from ultralytics import YOLO

    model = YOLO(model_name)
    model.train(**train_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
