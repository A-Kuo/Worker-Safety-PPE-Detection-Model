#!/usr/bin/env python3
"""Export a YOLO PPE checkpoint to ONNX (default 640)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _common import REPO_ROOT, ensure_repo_on_path  # noqa: E402
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
    ensure_repo_on_path()
    from ppe.tools.export import export_model  # the implementation ships in the package (ppe export)

    dest = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    target = "a *_openvino_model/ directory next to the source weights" if args.format == "openvino" else dest
    print(
        f"Export {args.format} from {weights} imgsz={args.imgsz} dynamic={args.dynamic} "
        f"half={args.half} opset={args.opset} -> {target}"
    )
    if args.dry_run:
        print(f"Dry-run: would write {target}")
        return 0
    written = export_model(
        weights,
        dest,
        imgsz=args.imgsz,
        dynamic=args.dynamic,
        simplify=args.simplify,
        half=args.half,
        opset=args.opset,
        fmt=args.format,
    )
    print(f"Wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
