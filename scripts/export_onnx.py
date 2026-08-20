#!/usr/bin/env python3
"""Export a YOLO checkpoint to an ONNX graph an NPU will accept.

NPU execution providers compile the graph ahead of time, which means fixed
input dimensions. This exports with static shapes by default; ``--dynamic`` is
available for CPU or GPU work but produces a model the strict NPU path will
refuse to load.

Needs torch, so this runs on a workstation rather than the device.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _common import REPO_ROOT, ensure_repo_on_path  # noqa: E402
from eval import resolve_weights  # noqa: E402

ensure_repo_on_path()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="export dynamic axes; NPU providers cannot compile these",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-simplify", dest="simplify", action="store_false")
    parser.add_argument("--half", action="store_true", help="FP16 weights; not for INT8 NPUs")
    parser.add_argument("--out", type=Path, default=None, help="destination .onnx path")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(simplify=True)
    return parser.parse_args()


def verify_static(path: Path) -> list[str]:
    """Report any dynamic input axes in the exported graph."""
    import onnxruntime

    from ppe.providers import dynamic_axes

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    problems = []
    for graph_input in session.get_inputs():
        axes = dynamic_axes(graph_input.shape)
        if axes:
            problems.append(f"{graph_input.name} has dynamic axes at {axes}: {graph_input.shape}")
    return problems


def main() -> int:
    args = _parse_args()
    weights = resolve_weights(args.weights)
    if weights is None:
        raise SystemExit("No weights found. Pass --weights path/to/best.pt")

    shape = "dynamic" if args.dynamic else f"1x3x{args.imgsz}x{args.imgsz}"
    print(f"Export {weights} to ONNX, opset {args.opset}, input {shape}")
    if args.dry_run:
        print(f"Dry run: would write {args.out or weights.with_suffix('.onnx')}")
        return 0

    from ultralytics import YOLO

    exported = Path(
        str(
            YOLO(str(weights)).export(
                format="onnx",
                imgsz=args.imgsz,
                dynamic=args.dynamic,
                simplify=args.simplify,
                half=args.half,
                opset=args.opset,
            )
        )
    )

    dest = exported
    if args.out:
        dest = args.out if args.out.is_absolute() else REPO_ROOT / args.out
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(exported.read_bytes())
    print(f"Wrote {dest}")

    problems = verify_static(dest)
    if problems and not args.dynamic:
        for line in problems:
            print(f"  warning: {line}", file=sys.stderr)
        print(
            "  The NPU path will reject this model. Re-export without --dynamic.",
            file=sys.stderr,
        )
        return 1
    if not problems:
        print("  Input shape is static; ready for NPU quantization.")
        print(f"  Next: python scripts/quantize_onnx.py --model {dest} --calibration data/calib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
