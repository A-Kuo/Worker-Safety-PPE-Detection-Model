"""``ppe`` console CLI — edge-oriented commands.

Examples (PowerShell — do not use angle brackets)::

    ppe providers
    ppe export --weights baselines/snehilsanyal_yolov8n_css/models/best.pt --out models/best.onnx
    ppe quantize --model models/best.onnx
    ppe bench --weights models/best.onnx --source path\\to\\image_or_video
    ppe predict --weights models/best.onnx --source path\\to\\frame.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_path() -> None:
    root = _repo_root()
    src = root / "src"
    for path in (str(root), str(src)):
        if path not in sys.path:
            sys.path.insert(0, path)


def cmd_providers(_args: argparse.Namespace) -> int:
    from ppe.runtime import describe_runtime, provider_status, resolve_providers

    rows = provider_status()
    print(json.dumps({"resolved": resolve_providers(None), "catalog": rows, "runtime": describe_runtime()}, indent=2))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    # Delegate to scripts/export_onnx.py logic via subprocess-free import path.
    root = _repo_root()
    scripts = root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from export_onnx import main as export_main  # type: ignore

    argv = ["export_onnx.py"]
    if args.weights:
        argv += ["--weights", str(args.weights)]
    if args.out:
        argv += ["--out", str(args.out)]
    argv += ["--imgsz", str(args.imgsz), "--format", args.format, "--opset", str(args.opset)]
    if args.dynamic:
        argv.append("--dynamic")
    old = sys.argv
    try:
        sys.argv = argv
        return int(export_main())
    finally:
        sys.argv = old


def cmd_quantize(args: argparse.Namespace) -> int:
    root = _repo_root()
    scripts = root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from quantize_onnx import main as quant_main  # type: ignore

    argv = ["quantize_onnx.py", "--model", str(args.model)]
    if args.out:
        argv += ["--out", str(args.out)]
    if args.calibration:
        argv += ["--calibration", str(args.calibration)]
    if args.static:
        argv.append("--static")
    old = sys.argv
    try:
        sys.argv = argv
        return int(quant_main())
    finally:
        sys.argv = old


def cmd_bench(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np

    from ppe.runtime import ExecutionPolicy, open_session

    root = _repo_root()
    policy = ExecutionPolicy.from_env(
        providers=tuple(args.providers.split(",")) if args.providers else (),
        npu_only=args.npu_only,
        allow_torch=args.allow_torch,
        imgsz=args.imgsz,
        conf=args.conf,
        openvino_device_type=args.openvino_device or None,
    )
    session = open_session(args.weights, policy=policy, repo_root=root)
    print(json.dumps(session.info(), indent=2, default=str))

    source = args.source
    if not source:
        # Synthetic frame if no source — still useful for latency.
        frames = [np.full((args.imgsz, args.imgsz, 3), 114, dtype=np.uint8)]
        label = "synthetic"
    else:
        src = Path(source)
        if not src.is_absolute():
            src = root / src
        if not src.exists():
            raise SystemExit(f"Source not found: {src}")
        if src.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            img = cv2.imread(str(src))
            if img is None:
                raise SystemExit(f"Failed to read image: {src}")
            frames = [img]
            label = str(src)
        else:
            cap = cv2.VideoCapture(str(src))
            if not cap.isOpened():
                raise SystemExit(f"Failed to open video/source: {src}")
            frames = []
            while len(frames) < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            cap.release()
            label = str(src)
            if not frames:
                raise SystemExit("No frames read from source")

    # Warmup
    for _ in range(args.warmup):
        session.predict(frames[0])

    latencies: list[float] = []
    total_dets = 0
    for frame in frames:
        t0 = time.perf_counter()
        dets = session.predict(frame)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        total_dets += len(dets)

    mean_ms = sum(latencies) / len(latencies)
    result = {
        "source": label,
        "frames": len(frames),
        "mean_latency_ms": round(mean_ms, 3),
        "fps": round(1000.0 / mean_ms, 2) if mean_ms else 0.0,
        "p50_ms": round(sorted(latencies)[len(latencies) // 2], 3),
        "detections_total": total_dets,
        "backend": session.backend.name,
        "model": str(session.model_path),
        "providers_active": session.backend.info().get("providers_active"),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = root / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {out}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    import cv2

    from ppe.runtime import ExecutionPolicy, open_session

    root = _repo_root()
    policy = ExecutionPolicy.from_env(
        providers=tuple(args.providers.split(",")) if args.providers else (),
        npu_only=args.npu_only,
        allow_torch=args.allow_torch,
        imgsz=args.imgsz,
        conf=args.conf,
        openvino_device_type=args.openvino_device or None,
    )
    session = open_session(args.weights, policy=policy, repo_root=root)
    src = Path(args.source)
    if not src.is_absolute():
        src = root / src
    image = cv2.imread(str(src))
    if image is None:
        raise SystemExit(f"Failed to read {src}")
    dets, workers = session.predict_and_comply(image)
    payload = {
        "model": str(session.model_path),
        "detections": [
            {"cls": d.cls_name, "conf": round(d.conf, 4), "xyxy": list(d.xyxy)} for d in dets
        ],
        "compliance": [
            {"worker_id": w.worker_id, "label": w.label, "missing": w.missing, "violations": w.violations}
            for w in workers
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ppe", description="PPE edge inference CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("providers", help="List ORT execution providers and availability")
    p.set_defaults(func=cmd_providers)

    p = sub.add_parser("export", help="Export YOLO weights to ONNX (static shapes by default)")
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("models/best.onnx"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--dynamic", action="store_true", help="Allow dynamic axes (worse for many NPUs)")
    p.add_argument("--format", choices=("onnx", "openvino"), default="onnx")
    p.add_argument("--opset", type=int, default=17)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("quantize", help="INT8-quantize an ONNX model")
    p.add_argument("--model", type=Path, default=Path("models/best.onnx"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--calibration", type=Path, default=None)
    p.add_argument("--static", action="store_true")
    p.set_defaults(func=cmd_quantize)

    p = sub.add_parser("bench", help="Latency / FPS bench on image, video, or synthetic frame")
    p.add_argument("--weights", type=Path, default=None, help="ONNX / INT8 / (opt-in) .pt")
    p.add_argument("--source", type=str, default=None, help="Image or video path (omit for synthetic)")
    p.add_argument("--providers", type=str, default="", help="Comma list, e.g. openvino,cpu")
    p.add_argument("--npu-only", action="store_true")
    p.add_argument("--allow-torch", action="store_true")
    p.add_argument(
        "--openvino-device",
        type=str,
        default="",
        help="OpenVINO EP device_type, e.g. AUTO:NPU,GPU / NPU / GPU / CPU",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--max-frames", type=int, default=100)
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("predict", help="Run one image through the edge session")
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--source", type=str, required=True)
    p.add_argument("--providers", type=str, default="")
    p.add_argument("--npu-only", action="store_true")
    p.add_argument("--allow-torch", action="store_true")
    p.add_argument(
        "--openvino-device",
        type=str,
        default="",
        help="OpenVINO EP device_type, e.g. AUTO:NPU,GPU / NPU / GPU / CPU",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.set_defaults(func=cmd_predict)

    return parser


def main(argv: list[str] | None = None) -> int:
    _ensure_path()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
