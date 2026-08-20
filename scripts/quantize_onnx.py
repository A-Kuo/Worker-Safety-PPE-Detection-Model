#!/usr/bin/env python3
"""Quantize an exported ONNX model to INT8 for NPU execution.

Hexagon, XDNA, and Ascend run integer kernels. Handed a float32 graph they
either refuse it or fall back to CPU node by node, which looks like the model
working and performing badly. Static QDQ quantization against real site frames
is what makes the accelerator actually take the graph.

Calibration wants a few hundred representative images. Frames from the cameras
the model will run on beat a random slice of the training set, because the
activation ranges that matter are the ones that occur on site.

Needs onnxruntime only, so this can run on the device or on a workstation.
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

DEFAULT_CALIBRATION_FRAMES = 200


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="float32 .onnx to quantize")
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="directory of representative frames, ideally from the target cameras",
    )
    parser.add_argument("--out", type=Path, default=None, help="destination .int8.onnx")
    parser.add_argument("--limit", type=int, default=DEFAULT_CALIBRATION_FRAMES)
    parser.add_argument(
        "--activation-type",
        choices=("uint8", "int8", "int16"),
        default="uint8",
        help="uint8 suits QNN and VitisAI; int16 helps when small objects degrade",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="per-channel weight scales; more accurate, not supported by every NPU",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


class FrameCalibrationReader:
    """Feeds letterboxed site frames to the calibrator.

    The preprocessing has to match :mod:`ppe.postprocess` exactly. Calibrating
    on differently scaled inputs produces activation ranges the runtime never
    sees, and the quantized model loses accuracy for no visible reason.
    """

    def __init__(self, images: list[Path], input_name: str, imgsz: int) -> None:
        from ppe.postprocess import letterbox, to_input_tensor
        from ppe.sources import read_image

        self.input_name = input_name
        self._batches = iter(
            [
                {input_name: to_input_tensor(letterbox(read_image(path), imgsz)[0])}
                for path in images
            ]
        )

    def get_next(self):
        return next(self._batches, None)

    def rewind(self) -> None:  # pragma: no cover - onnxruntime calls this only sometimes
        raise NotImplementedError("Build a fresh reader instead of rewinding")


def collect_frames(root: Path, limit: int) -> list[Path]:
    from ppe.sources import IMAGE_SUFFIXES

    if not root.is_dir():
        raise SystemExit(f"Calibration directory not found: {root}")
    frames = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not frames:
        raise SystemExit(f"No images under {root}. Calibration needs real frames.")
    if len(frames) < 50:
        print(
            f"  warning: only {len(frames)} calibration frames. "
            "Under about 50 the activation ranges are unreliable.",
            file=sys.stderr,
        )
    return frames[:limit]


def model_input(path: Path) -> tuple[str, int]:
    import onnxruntime

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    graph_input = session.get_inputs()[0]
    shape = graph_input.shape
    height = shape[2] if len(shape) > 2 else None
    if not isinstance(height, int) or height <= 0:
        raise SystemExit(
            f"{path.name} has a dynamic input shape {shape}. Re-export with static dimensions:\n"
            "  python scripts/export_onnx.py --weights best.pt --imgsz 640"
        )
    return graph_input.name, height


def main() -> int:
    args = _parse_args()
    model = args.model if args.model.is_absolute() else REPO_ROOT / args.model
    if not model.is_file():
        raise SystemExit(f"Model not found: {model}")

    out = args.out or model.with_suffix(".int8.onnx")
    out = out if out.is_absolute() else REPO_ROOT / out

    input_name, imgsz = model_input(model)
    frames = collect_frames(args.calibration, args.limit)
    print(f"Quantize {model.name} to INT8 using {len(frames)} frames at {imgsz}x{imgsz}")

    if args.dry_run:
        print(f"Dry run: would write {out}")
        return 0

    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    activation = {
        "uint8": QuantType.QUInt8,
        "int8": QuantType.QInt8,
        "int16": QuantType.QUInt16,
    }[args.activation_type]

    out.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(model),
        model_output=str(out),
        calibration_data_reader=FrameCalibrationReader(frames, input_name, imgsz),
        quant_format=QuantFormat.QDQ,
        activation_type=activation,
        weight_type=QuantType.QInt8,
        per_channel=args.per_channel,
    )

    before = model.stat().st_size / 1e6
    after = out.stat().st_size / 1e6
    print(f"Wrote {out} ({before:.1f} MB to {after:.1f} MB)")
    print("  Verify accuracy before deploying: quantization moves mAP, sometimes a lot.")
    print(f"  python scripts/eval.py --weights {out}")
    print(f"  ppe bench --weights {out} --execution npu --json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
