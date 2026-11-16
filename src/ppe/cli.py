"""``ppe`` console CLI — edge-oriented commands.

Examples (PowerShell — do not use angle brackets)::

    ppe providers
    ppe export --weights runs/train/vest_specialist/weights/best.pt --out models/vest_specialist.onnx
    ppe quantize --model models/best.onnx
    ppe verify-npu --weights models/best.onnx --providers openvino
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
    """Base for relative paths: PPE_HOME, else the source checkout, else the CWD (wheel install)."""
    from ppe.paths import project_root

    return project_root()


def _abs(path: Path | str, root: Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else root / path


def _ensure_path() -> None:
    # Only a source checkout needs (and may safely get) its root/src on sys.path. From a wheel the
    # project root is the CWD, and putting an arbitrary CWD on the import path would be wrong.
    from ppe.paths import source_root

    root = source_root()
    if root is None:
        return
    for path in (str(root), str(root / "src")):
        if path not in sys.path:
            sys.path.insert(0, path)


def cmd_providers(_args: argparse.Namespace) -> int:
    from ppe.runtime import describe_runtime, provider_status, resolve_providers

    rows = provider_status()
    print(json.dumps({"resolved": resolve_providers(None), "catalog": rows, "runtime": describe_runtime()}, indent=2))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from ppe.tools.export import export_model

    root = _repo_root()
    if not args.weights:
        raise SystemExit("ppe export: pass --weights path/to/best.pt")
    try:
        written = export_model(
            _abs(args.weights, root),
            _abs(args.out, root),
            imgsz=args.imgsz,
            dynamic=args.dynamic,
            opset=args.opset,
            fmt=args.format,
        )
    except (RuntimeError, FileNotFoundError) as exc:
        raise SystemExit(f"ppe export: {exc}") from exc
    print(f"Wrote {written}")
    return 0


def cmd_quantize(args: argparse.Namespace) -> int:
    from ppe.tools.quantize import quantize_model

    root = _repo_root()
    try:
        written = quantize_model(
            _abs(args.model, root),
            _abs(args.out, root) if args.out else None,
            calibration=_abs(args.calibration, root) if args.calibration else None,
            static=args.static,
            qdq=args.qdq,
            per_channel=args.per_channel,
            limit=args.calibration_images,
            seed=args.seed,
            float_head=args.float_head,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"ppe quantize: {exc}") from exc
    print(f"Wrote {written} ({written.stat().st_size} bytes)")
    print("Note: validate accuracy after INT8 quantization before using the model.")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np

    from ppe.runtime import open_session
    from ppe.runtime.fingerprint import scrub_paths

    root = _repo_root()
    policy = _policy_from_args(args)
    session = open_session(args.weights, policy=policy, repo_root=root)

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

    from ppe.runtime.stats import fps_from_ms, summarize, summarize_stages

    # Warmup: the first runs include NPU graph compilation / kernel autotune and must not be timed.
    for i in range(args.warmup):
        session.predict(frames[i % len(frames)])

    iterations = max(1, args.iterations)
    totals: list[float] = []
    stage_samples: list[dict[str, float]] = []
    total_dets = 0
    wall0 = time.perf_counter()
    for i in range(iterations):
        t0 = time.perf_counter()
        dets = session.predict(frames[i % len(frames)])
        totals.append((time.perf_counter() - t0) * 1000.0)
        stage_samples.append(session.last_timings)
        total_dets += len(dets)
    wall_s = time.perf_counter() - wall0

    latency = summarize(totals)
    info = session.backend.info()
    result = {
        "source": label,
        "source_frames": len(frames),
        "iterations": iterations,
        "warmup": args.warmup,
        "latency_ms": latency,
        "fps_mean": round(iterations / wall_s, 2) if wall_s > 0 else 0.0,
        "fps_from_p50": fps_from_ms(latency.get("p50", 0.0)),
        "fps_from_p95": fps_from_ms(latency.get("p95", 0.0)),
        "stages_ms": summarize_stages(stage_samples),
        "detections_total": total_dets,
        "backend": session.backend.name,
        "model": str(session.model_path),
        "specialists": [str(s.path) for s in session.specialist_backends],
        "imgsz": args.imgsz,
        "tag": args.tag or None,
        "strict_npu": session.policy.strict_npu,
        "providers_active": info.get("providers_active"),
        "openvino_device_type": info.get("openvino_device_type"),
        "fingerprint": info.get("fingerprint"),
    }
    if iterations < 100:
        result["note"] = "fewer than 100 iterations: treat p95/p99 as indicative only"
    result = scrub_paths(result, root)
    print(json.dumps(result, indent=2))
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = root / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {out}")
    return 0


def _policy_from_args(args: argparse.Namespace) -> "ExecutionPolicy":
    from ppe.runtime import ExecutionPolicy
    from ppe.runtime.config import _parse_specialist_paths

    # Only override env when the flag was actually passed — empty --providers
    # must not wipe PPE_PROVIDERS / OpenVINO NPU pins from the environment.
    overrides: dict = {
        "imgsz": args.imgsz,
        "conf": args.conf,
    }
    if args.providers:
        overrides["providers"] = tuple(p.strip() for p in args.providers.split(",") if p.strip())
    if args.npu_only:
        overrides["npu_only"] = True
    # None = flag not given (keep PPE_STRICT_NPU from the env); True/False = explicit, so
    # `verify-npu --soft` really does override an env-enabled strict mode.
    strict = getattr(args, "strict_npu", None)
    if strict is not None:
        overrides["strict_npu"] = bool(strict)
    if args.allow_torch:
        overrides["allow_torch"] = True
    if getattr(args, "openvino_device", None):
        overrides["openvino_device_type"] = args.openvino_device
    specialists = _parse_specialist_paths(getattr(args, "specialists", "") or None, None)
    if specialists:
        overrides["specialists"] = specialists
    return ExecutionPolicy.from_env(**overrides)


def cmd_verify_npu(args: argparse.Namespace) -> int:
    """Prove which execution provider ran every node. Exit 0 only if nothing ran on the CPU."""
    from ppe.runtime.config import resolve_model_path
    from ppe.runtime.fingerprint import scrub_paths
    from ppe.runtime.placement import verify_placement

    root = _repo_root()
    args.strict_npu = not args.soft  # explicit either way: --soft overrides PPE_STRICT_NPU
    policy = _policy_from_args(args)
    try:
        model = resolve_model_path(args.weights, prefer_onnx=True, repo_root=root)
        report = verify_placement(model, policy, runs=args.runs)
    except Exception as exc:  # noqa: BLE001 - a diagnostic must always emit machine-readable JSON
        # Strict refusals (no NPU EP, dynamic shape, CPU-resident op), a missing model, or an ORT
        # failure are *results* for scripts and CI to parse, not tracebacks.
        err = {"ok": False, "strict_npu": policy.strict_npu, "error_type": type(exc).__name__,
               "error": str(exc)}
        print(json.dumps(scrub_paths(err, root), indent=2))
        return 1
    report = scrub_paths(report, root)  # committed evidence must not carry local absolute paths
    print(json.dumps(report, indent=2))
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = root / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {out}")
    return 0 if report["ok"] else 1


def cmd_predict(args: argparse.Namespace) -> int:
    import cv2

    from ppe.runtime import open_session

    root = _repo_root()
    policy = _policy_from_args(args)
    session = open_session(args.weights, policy=policy, repo_root=root)
    src = Path(args.source)
    if not src.is_absolute():
        src = root / src
    image = cv2.imread(str(src))
    if image is None:
        raise SystemExit(f"Failed to read {src}")
    dets, workers = session.predict_and_comply(image)
    counts: dict[str, int] = {}
    for d in dets:
        counts[d.cls_name] = counts.get(d.cls_name, 0) + 1
    payload = {
        "model": str(session.model_path),
        # Frame-level result: what a model without a `person` class (e.g. the 2-class vest
        # specialist) can report. Per-worker compliance below needs person boxes to attach PPE to.
        "counts": dict(sorted(counts.items())),
        "detections": [
            {"cls": d.cls_name, "conf": round(d.conf, 4), "xyxy": list(d.xyxy)} for d in dets
        ],
        "compliance": [
            {"worker_id": w.worker_id, "label": w.label, "missing": w.missing, "violations": w.violations}
            for w in workers
        ],
    }
    if not _model_classes(session) & {"person"}:
        payload["compliance_note"] = (
            "This model has no `person` class, so per-worker compliance is empty by design; "
            "use `counts` / `detections` (frame-level)."
        )
    print(json.dumps(payload, indent=2))
    return 0


def _model_classes(session) -> set[str]:
    """Unified class names the session's models can emit."""
    from ppe.runtime.backends import _to_unified

    names: set[str] = set()
    for backend in (session.backend, *(s.backend for s in session.specialist_backends)):
        raw = getattr(backend, "class_names", None) or {}
        names.update(_to_unified(str(v)) for v in raw.values())
    return names


def cmd_eval(args: argparse.Namespace) -> int:
    """Score the shipped runtime path on a labeled image folder (YOLO labels), with counts and CIs."""
    from ppe.evaluate import evaluate_session, list_images
    from ppe.runtime import open_session
    from ppe.runtime.fingerprint import scrub_paths

    root = _repo_root()
    args.conf = args.conf_floor
    policy = _policy_from_args(args)
    policy.class_conf = {}  # score the raw model: no per-class thresholds from the environment
    session = open_session(args.weights, policy=policy, repo_root=root)
    images_dir = _abs(args.images, root)
    images = list_images(images_dir)
    if not images:
        raise SystemExit(f"ppe eval: no images under {images_dir}")
    if args.names:
        names = [n.strip() for n in args.names.split(",") if n.strip()]
    else:
        names = [session.backend.class_names[i] for i in sorted(session.backend.class_names)]
    summary = evaluate_session(session, images, names, match_iou=args.match_iou)
    info = session.backend.info()
    result = {
        "model": str(session.model_path),
        "images_dir": str(images_dir),
        "label_names": names,
        "tag": args.tag or None,
        "protocol": {
            "path": "ppe runtime (ORT decode + class-agnostic NMS IoU 0.45)",
            "conf_floor": args.conf_floor,
            "match_iou": args.match_iou,
            "ap": "AP50, 101-point interpolation",
            "intervals": "Wilson 95%",
        },
        **summary,
        "strict_npu": session.policy.strict_npu,
        "providers_active": info.get("providers_active"),
        "openvino_device_type": info.get("openvino_device_type"),
        "fingerprint": info.get("fingerprint"),
    }
    result = scrub_paths(result, root)
    brief = {
        n: {"n_gt_boxes": c["n_gt_boxes"], "ap50": c["ap50"]} for n, c in result["per_class"].items()
    }
    print(json.dumps({"n_images": result["n_images"], "map50": result["map50"], "per_class": brief}, indent=2))
    if args.out:
        out = _abs(args.out, root)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {out}")
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
    p.add_argument("--qdq", action="store_true", help="Static only: QDQ format, uint8 activations (NPU-friendly)")
    p.add_argument("--per-channel", action="store_true", help="Static only: per-channel weight scales")
    p.add_argument(
        "--float-head",
        action="store_true",
        help="Static only: keep the YOLO decode head in float (required: a fully quantized head "
        "zeroes class scores on some devices)",
    )
    p.add_argument("--calibration-images", type=int, default=100)
    p.add_argument("--seed", type=int, default=None, help="Random but reproducible calibration sample")
    p.set_defaults(func=cmd_quantize)

    p = sub.add_parser("bench", help="Latency / FPS bench on image, video, or synthetic frame")
    p.add_argument("--weights", type=Path, default=None, help="ONNX / INT8 / (opt-in) .pt")
    p.add_argument("--source", type=str, default=None, help="Image or video path (omit for synthetic)")
    p.add_argument("--providers", type=str, default="", help="Comma list, e.g. openvino,cpu")
    p.add_argument("--npu-only", action="store_true")
    p.add_argument("--strict-npu", action="store_true", default=None, help="No CPU EP, no CPU fallback, static shapes")
    p.add_argument("--allow-torch", action="store_true")
    p.add_argument(
        "--openvino-device",
        type=str,
        default="",
        help="OpenVINO EP device_type, e.g. AUTO:NPU,GPU / NPU / GPU / CPU",
    )
    p.add_argument(
        "--specialists",
        type=str,
        default="",
        help="Comma list of class-pair specialist ONNX/.pt paths (vest and/or helmet)",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--warmup", type=int, default=20, help="Untimed runs (NPU compile/autotune)")
    p.add_argument("--iterations", type=int, default=300, help="Timed predictions (frames are cycled)")
    p.add_argument("--max-frames", type=int, default=100, help="Max frames read from a video source")
    p.add_argument("--tag", type=str, default="", help="Free-form label stored in the result")
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser(
        "verify-npu",
        help="Profile one run and prove which provider executed every node (exit 1 if any ran on CPU)",
    )
    p.add_argument("--weights", type=Path, default=None, help="ONNX / INT8 model")
    p.add_argument("--providers", type=str, default="", help="Comma list, e.g. openvino")
    p.add_argument("--npu-only", action="store_true")
    p.add_argument("--allow-torch", action="store_true")
    p.add_argument("--openvino-device", type=str, default="", help="Default under strict mode: NPU")
    p.add_argument("--specialists", type=str, default="")
    p.add_argument(
        "--soft",
        action="store_true",
        help="Diagnose without strict enforcement: list CPU-resident ops instead of refusing",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--runs", type=int, default=3, help="Profiled inferences on a zero tensor")
    p.add_argument("--out", type=Path, default=None, help="Write the JSON report here")
    p.set_defaults(func=cmd_verify_npu)

    p = sub.add_parser("eval", help="Score the runtime on a labeled image folder (counts, AP50, Wilson CIs)")
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--images", type=Path, required=True, help="Folder of images; labels in sibling labels/")
    p.add_argument("--names", type=str, default="", help="Label class names in id order (default: model's)")
    p.add_argument("--providers", type=str, default="")
    p.add_argument("--npu-only", action="store_true")
    p.add_argument("--strict-npu", action="store_true", default=None)
    p.add_argument("--allow-torch", action="store_true")
    p.add_argument("--openvino-device", type=str, default="")
    p.add_argument("--specialists", type=str, default="")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf-floor", type=float, default=0.001, help="Lowest confidence kept (mAP protocol)")
    p.add_argument("--match-iou", type=float, default=0.5)
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("predict", help="Run one image through the edge session")
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--source", type=str, required=True)
    p.add_argument("--providers", type=str, default="")
    p.add_argument("--npu-only", action="store_true")
    p.add_argument("--strict-npu", action="store_true", default=None, help="No CPU EP, no CPU fallback, static shapes")
    p.add_argument("--allow-torch", action="store_true")
    p.add_argument(
        "--openvino-device",
        type=str,
        default="",
        help="OpenVINO EP device_type, e.g. AUTO:NPU,GPU / NPU / GPU / CPU",
    )
    p.add_argument(
        "--specialists",
        type=str,
        default="",
        help="Comma list of class-pair specialist ONNX/.pt paths (vest and/or helmet)",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.set_defaults(func=cmd_predict)

    return parser


def main(argv: list[str] | None = None) -> int:
    _ensure_path()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        # A missing model/source is a usage problem: one line, not a traceback.
        raise SystemExit(f"ppe {args.command}: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
