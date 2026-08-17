#!/usr/bin/env python3
"""Export a YOLO PPE checkpoint to ONNX (default 640)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _common import REPO_ROOT  # noqa: E402
from eval import resolve_weights  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--simplify", action="store_true", default=True)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--out", type=Path, default=None, help="Optional destination .onnx path")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    weights = resolve_weights(args.weights)
    if weights is None:
        raise SystemExit("No weights found. Pass --weights path/to/best.pt")

    print(f"Export ONNX from {weights} imgsz={args.imgsz} dynamic={args.dynamic}")
    if args.dry_run:
        dest = args.out or weights.with_suffix(".onnx")
        print(f"Dry-run: would write {dest}")
        return 0

    from ultralytics import YOLO

    model = YOLO(str(weights))
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        dynamic=args.dynamic,
        simplify=args.simplify,
        half=args.half,
    )
    exported_path = Path(str(exported))
    if args.out:
        dest = args.out if args.out.is_absolute() else REPO_ROOT / args.out
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(exported_path.read_bytes())
        print(f"Wrote {dest}")
    else:
        print(f"Wrote {exported_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
