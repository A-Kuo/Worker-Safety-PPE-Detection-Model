#!/usr/bin/env python3
"""CLI wrapper around Ultralytics ``YOLO.train`` for the E0–E4 configs.

Reads ``configs/train/{e0_n,e1_s,e2_focal,e3_augs,e4_full44k,vest_specialist}.yaml``.
Resume-friendly. ``--dry-run`` prints the resolved kwargs and exits - this
script does not start a run unless you omit ``--dry-run``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _common import REPO_ROOT, first_existing, load_yaml, validate_train_kwargs  # noqa: E402

EXPERIMENTS = ("e0_n", "e1_s", "e2_focal", "e3_augs", "e4_full44k", "vest_specialist")
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
        "--fraction",
        type=float,
        default=None,
        help="Train on only this fraction of the train set (smoke tests: 0.02).",
    )
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


def _subset_dirname(n: int) -> str:
    """Match scripts/make_subset.py's actual --out convention (12000 -> "combined_12k")."""
    if n % 1000 == 0:
        return f"combined_{n // 1000}k"
    return f"combined_{n}"


def verify_subset_paths(subset_yaml: Path) -> None:
    """Fail if a subset's image lists point outside the dataset it was built from.

    Ultralytics derives each label path from the listed image path, so a list that
    escaped into raw/ (e.g. via a resolved symlink) trains on unremapped labels.
    """
    manifest_path = subset_yaml.parent / "subset_manifest.json"
    if not manifest_path.exists():
        return
    source = json.loads(manifest_path.read_text(encoding="utf-8")).get("source")
    if not source:
        return
    prefixes = {
        os.path.normcase(os.path.abspath(source)) + os.sep,
        os.path.normcase(os.path.realpath(source)) + os.sep,
    }
    for name in ("train.txt", "valid.txt", "test.txt"):
        list_path = subset_yaml.parent / name
        if not list_path.exists():
            continue
        with list_path.open(encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                if i >= 500:
                    break
                entry = line.strip()
                if entry and not any(os.path.normcase(os.path.abspath(entry)).startswith(p) for p in prefixes):
                    raise SystemExit(
                        f"{list_path} lists {entry}, which is outside the subset's recorded source "
                        f"({source}). Its labels would not be the source dataset's remapped labels. "
                        "Rebuild it: python scripts/make_subset.py --source data/processed/combined "
                        f"--out {subset_yaml.parent}"
                    )


def resolve_subset_data(subset_size: int) -> Path:
    candidate = REPO_ROOT / "data" / "raw" / _subset_dirname(subset_size) / "data.yaml"
    if not candidate.exists():
        raise SystemExit(
            f"subset_size={subset_size} is set in this config but {candidate} does not "
            "exist yet. Build it first:\n"
            f"  python scripts/make_subset.py --source data/processed/combined "
            f"--out data/raw/{_subset_dirname(subset_size)} --n {subset_size}"
        )
    verify_subset_paths(candidate)
    return candidate


def load_train_kwargs(path: Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    # Allow a nested `train:` block or a flat Ultralytics kwargs file.
    if isinstance(cfg.get("train"), dict) and "data" not in cfg:
        cfg = dict(cfg["train"])
    subset_size = cfg.get("subset_size")
    for key in META_KEYS:
        cfg.pop(key, None)
    if subset_size:
        # subset_size takes precedence over `data:` — previously this key was
        # silently dropped and every experiment trained on the full dataset
        # regardless of what subset_size said, contradicting docs/experiments.md's
        # "isolate on 12k, confirm once on 44k" protocol.
        cfg["data"] = str(resolve_subset_data(int(subset_size)))
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
    if args.fraction is not None:
        cfg["fraction"] = args.fraction
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

    validate_train_kwargs(train_kwargs)

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
    # On Kaggle, stage the weights into /kaggle/working/deliverables_<name> right away (no-op elsewhere).
    from package_deliverables import auto_package  # noqa: E402

    auto_package(str(train_kwargs.get("name") or args.exp), REPO_ROOT, prune_after=False, strict=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
