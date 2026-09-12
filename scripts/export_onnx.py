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
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset (ignored for --format openvino)")
    parser.add_argument(
        "--format",
        choices=("onnx", "openvino"),
        default="onnx",
        help="onnx: plain ONNX, run through any ORT EP. "
        "openvino: Ultralytics' native OpenVINO IR export (a *_openvino_model/ dir), "
        "an alternative to running plain ONNX through the OpenVINO EP.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/best.onnx"),
        help="Destination path (default: models/best.onnx; ignored for --format openvino, "
        "which always writes next to the source weights). Static shapes preferred for NPU.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    weights = resolve_weights(args.weights)
    if weights is None:
        raise SystemExit("No weights found. Pass --weights path/to/best.pt")

    if args.format == "openvino":
        print(
            f"Export OpenVINO IR from {weights} imgsz={args.imgsz} "
            f"dynamic={args.dynamic} half={args.half}"
        )
        if args.dry_run:
            print("Dry-run: would write a *_openvino_model/ directory next to the source weights")
            return 0
        from ultralytics import YOLO

        model = YOLO(str(weights))
        exported = model.export(
            format="openvino",
            imgsz=args.imgsz,
            dynamic=args.dynamic,
            half=args.half,
        )
        print(f"Wrote {exported}")
        return 0

    dest = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    print(
        f"Export ONNX from {weights} imgsz={args.imgsz} dynamic={args.dynamic} "
        f"opset={args.opset} -> {dest}"
    )
    if args.dry_run:
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
        opset=args.opset,
    )
    exported_path = Path(str(exported))
    dest.parent.mkdir(parents=True, exist_ok=True)
    if exported_path.resolve() != dest.resolve():
        dest.write_bytes(exported_path.read_bytes())
    print(f"Wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
