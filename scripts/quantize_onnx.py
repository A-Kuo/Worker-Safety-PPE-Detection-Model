#!/usr/bin/env python3
"""INT8-quantize an ONNX PPE model for edge / NPU deployment.

Default: **dynamic** quantization (no calibration images required).
Optional: **static** quantization when ``--calibration`` points at an image folder.

Examples::

    python scripts/quantize_onnx.py --model models/best.onnx
    python scripts/quantize_onnx.py --model models/best.onnx --calibration data/calib --static
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _common import REPO_ROOT, ensure_repo_on_path  # noqa: E402

ensure_repo_on_path()
# The implementation ships in the package (``ppe quantize``); this script adds repo-relative defaults.
from ppe.tools.quantize import quantize_dynamic, quantize_static  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=REPO_ROOT / "models" / "best.onnx",
        help="Input ONNX (FP32). Default: models/best.onnx",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output INT8 ONNX. Default: <model_stem>.int8.onnx next to input or under models/",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Folder of images for static quantization (optional).",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Use static quantization (requires --calibration).",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exit without writing.",
    )
    return parser.parse_args()


def _resolve_out(model: Path, out: Path | None) -> Path:
    if out:
        path = out if out.is_absolute() else REPO_ROOT / out
        return path
    if model.parent.name == "models" or model.parent == REPO_ROOT / "models":
        return model.with_name(model.stem + ".int8.onnx")
    dest_dir = REPO_ROOT / "models"
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir / (model.stem + ".int8.onnx")


def main() -> int:
    args = _parse_args()
    model = args.model if args.model.is_absolute() else REPO_ROOT / args.model
    if not model.is_file():
        raise SystemExit(
            f"ONNX model not found: {model}\n"
            "Export first, e.g.\n"
            "  python scripts/export_onnx.py "
            "--weights archive/baselines/snehilsanyal_yolov8n_css/models/best.pt "
            "--out models/best.onnx"
        )
    out = _resolve_out(model, args.out)
    mode = "static" if args.static else "dynamic"
    if args.static and not args.calibration:
        raise SystemExit("--static requires --calibration <image_folder>")

    print(f"Quantize ({mode}): {model} -> {out}")
    if args.calibration:
        print(f"Calibration: {args.calibration}")
    if args.dry_run:
        print("Dry-run: not writing")
        return 0

    if args.static:
        calib = args.calibration if args.calibration.is_absolute() else REPO_ROOT / args.calibration
        quantize_static(model, out, calib, args.imgsz)
    else:
        quantize_dynamic(model, out)
    print(f"Wrote {out} ({out.stat().st_size} bytes)")
    print(
        "Note: INT8 dynamic quant is a best-effort edge appendix. "
        "Validate mAP before shipping; static calib usually recovers more accuracy."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
