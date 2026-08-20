"""Command line entry point: ``python -m ppe`` or the installed ``ppe`` script.

Subcommands
    ``devices``  list NPU execution providers and whether this host has them
    ``image``    score stills, optionally writing annotated copies
    ``video``    score a clip and print a run summary
    ``watch``    follow a camera or RTSP stream and print alerts as they raise
    ``bench``    measure latency and throughput for the configured backend
    ``classes``  print the unified label schema
    ``serve``    start the FastAPI service
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ppe import __version__
from ppe.config import BACKENDS, RuntimeConfig, config_from_env
from ppe.pipeline import EdgePipeline, summarize_run
from ppe.providers import EXECUTION_POLICIES, describe_environment
from ppe.schema import UNIFIED_CLASS_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ppe", description="PPE compliance detection.")
    parser.add_argument("--version", action="version", version=f"ppe {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    image = subparsers.add_parser("image", help="score one image or a directory of images")
    image.add_argument("path", help="image file or directory")
    image.add_argument("--save", metavar="DIR", help="write annotated copies here")
    _add_model_args(image)
    _add_output_args(image)
    image.set_defaults(handler=cmd_image)

    video = subparsers.add_parser("video", help="score a video file")
    video.add_argument("path", help="video file")
    video.add_argument("--save", metavar="MP4", help="write an annotated clip here")
    video.add_argument("--max-frames", type=int, default=300)
    video.add_argument("--max-seconds", type=float, default=45.0)
    _add_model_args(video)
    _add_output_args(video)
    video.set_defaults(handler=cmd_video)

    watch = subparsers.add_parser("watch", help="follow a camera or stream and print alerts")
    watch.add_argument("source", nargs="?", default="0", help="camera index, RTSP URL, or file")
    watch.add_argument(
        "--seconds", type=float, default=0.0, help="stop after N seconds (0 = never)"
    )
    watch.add_argument("--report-every", type=float, default=10.0, help="latency report interval")
    _add_model_args(watch)
    _add_output_args(watch)
    watch.set_defaults(handler=cmd_watch)

    bench = subparsers.add_parser("bench", help="measure latency and throughput")
    bench.add_argument("--source", help="optional video or image to benchmark against")
    bench.add_argument("--frames", type=int, default=100)
    bench.add_argument("--warmup", type=int, default=5)
    _add_model_args(bench)
    _add_output_args(bench)
    bench.set_defaults(handler=cmd_bench)

    classes = subparsers.add_parser("classes", help="print the unified label schema")
    _add_output_args(classes)
    classes.set_defaults(handler=cmd_classes)

    devices = subparsers.add_parser("devices", help="list NPU execution providers on this host")
    _add_output_args(devices)
    devices.set_defaults(handler=cmd_devices)

    serve = subparsers.add_parser("serve", help="start the FastAPI service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(handler=cmd_serve)

    return parser


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("model")
    group.add_argument("--weights", help="ONNX model path; defaults to PPE_WEIGHTS")
    group.add_argument(
        "--backend",
        choices=BACKENDS,
        default=None,
        help="auto resolves to onnx; ultralytics also needs PPE_ALLOW_TORCH=1",
    )
    group.add_argument(
        "--execution",
        choices=EXECUTION_POLICIES,
        default=None,
        help="npu fails when no NPU provider is present; npu-preferred falls back to CPU",
    )
    group.add_argument("--provider", help="pin one execution provider, e.g. QNNExecutionProvider")
    group.add_argument(
        "--provider-option",
        action="append",
        metavar="KEY=VALUE",
        help="provider option override; repeatable",
    )
    group.add_argument("--device", help="torch device for the reference backend only")
    group.add_argument("--conf", type=float, help="confidence threshold")
    group.add_argument("--iou", type=float, help="NMS IoU threshold")
    group.add_argument("--imgsz", type=int, help="inference size, a multiple of 32")
    group.add_argument("--required", help="comma-separated PPE every worker must wear")
    group.add_argument("--stride", type=int, help="process every Nth frame")
    group.add_argument("--alert-frames", type=int, help="consecutive frames before an alert")
    group.add_argument("--alert-cooldown", type=float, help="seconds between repeat alerts")


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")


def config_from_args(args: argparse.Namespace) -> RuntimeConfig:
    """Environment first, command line on top."""
    config = config_from_env()
    required = getattr(args, "required", None)
    options = _parse_provider_options(getattr(args, "provider_option", None))
    return config.with_overrides(
        weights=Path(args.weights).expanduser() if getattr(args, "weights", None) else None,
        backend=getattr(args, "backend", None),
        execution=getattr(args, "execution", None),
        provider=getattr(args, "provider", None),
        provider_options={**config.provider_options, **options} if options else None,
        device=getattr(args, "device", None),
        conf=getattr(args, "conf", None),
        iou=getattr(args, "iou", None),
        imgsz=getattr(args, "imgsz", None),
        required_ppe=tuple(p.strip() for p in required.split(",") if p.strip())
        if required
        else None,
        frame_stride=getattr(args, "stride", None),
        alert_min_frames=getattr(args, "alert_frames", None),
        alert_cooldown_s=getattr(args, "alert_cooldown", None),
    )


def _parse_provider_options(pairs: list[str] | None) -> dict[str, str]:
    options: dict[str, str] = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"--provider-option expects KEY=VALUE, got {pair!r}")
        options[key.strip()] = value.strip()
    return options


def cmd_devices(args: argparse.Namespace) -> int:
    """Report which accelerators this host can actually bind."""
    env = describe_environment()
    if args.json:
        _emit(env)
        return 0 if env["npu_available"] else 1

    print(f"onnxruntime {env['onnxruntime']} on {env['platform']}")
    print(f"installed providers: {', '.join(env['installed_providers'])}")
    print()
    width = max(len(row["provider"]) for row in env["providers"])
    for row in env["providers"]:
        mark = "yes" if row["available"] else "no "
        print(f"  [{mark}] {row['provider']:<{width}}  {row['vendor']} {row['hardware']}")
        if not row["available"]:
            print(f"        {row['install']}")
    print()

    if env["npu_available"]:
        print(f"NPU ready: {', '.join(env['npu_available'])}")
        return 0
    print(
        "No NPU execution provider is installed, so --execution npu will refuse to start.\n"
        "Install one of the builds above, or pass --execution cpu to accept CPU inference."
    )
    return 1


def cmd_classes(args: argparse.Namespace) -> int:
    if args.json:
        _emit({"classes": UNIFIED_CLASS_NAMES, "count": len(UNIFIED_CLASS_NAMES)})
    else:
        for index, name in enumerate(UNIFIED_CLASS_NAMES):
            print(f"{index:2d}  {name}")
    return 0


def cmd_image(args: argparse.Namespace) -> int:
    from ppe.draw import annotate
    from ppe.sources import iter_images

    config = config_from_args(args)
    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    payload = []
    with EdgePipeline.from_config(config) as pipeline:
        for path, frame in iter_images(args.path):
            pipeline.reset()
            result = pipeline.process(frame)
            record = {"image": str(path), **result.as_dict()}
            if save_dir:
                import cv2

                out_path = save_dir / f"{path.stem}_annotated{path.suffix or '.jpg'}"
                cv2.imwrite(str(out_path), annotate(frame, result.detections, result.workers))
                record["annotated"] = str(out_path)
            payload.append(record)
            if not args.json:
                print(f"{path}: {_describe(result)}")

    if not payload:
        print(f"No images found under {args.path}", file=sys.stderr)
        return 1
    if args.json:
        _emit(payload)
    return 0


def cmd_video(args: argparse.Namespace) -> int:
    from ppe.draw import annotate
    from ppe.sources import capture_info, frame_stride, iter_capture, open_capture, open_writer

    config = config_from_args(args)
    cap = open_capture(args.path)
    info = capture_info(cap, args.path)
    stride = frame_stride(info.frame_count, args.max_frames)
    writer = None
    results = []

    try:
        with EdgePipeline.from_config(config) as pipeline:
            for _index, frame, timestamp in iter_capture(
                cap,
                max_frames=args.max_frames,
                max_seconds=args.max_seconds,
                fps=info.fps,
                stride=stride,
            ):
                result = pipeline.process(frame, timestamp)
                results.append(result)
                if args.save:
                    if writer is None:
                        height, width = frame.shape[:2]
                        writer = open_writer(args.save, info.fps / stride, (width, height))
                    writer.write(annotate(frame, result.detections, result.workers))
            report = {
                "source": args.path,
                "fps": info.fps,
                "frame_count": info.frame_count,
                "stride": stride,
                **summarize_run(results),
                "latency": pipeline.latency.as_dict(),
            }
    finally:
        if writer is not None:
            writer.release()
        cap.release()

    if args.save:
        report["annotated"] = args.save
    if args.json:
        _emit(report)
    else:
        _print_report(report)
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    from ppe.sources import capture_info, iter_capture, open_capture

    config = config_from_args(args)
    cap = open_capture(args.source)
    info = capture_info(cap, args.source)
    deadline = time.monotonic() + args.seconds if args.seconds > 0 else None
    next_report = time.monotonic() + args.report_every if args.report_every > 0 else None

    print(f"Watching {info.spec} ({info.kind}, {info.fps:.1f} fps). Ctrl-C to stop.")
    try:
        with EdgePipeline.from_config(config) as pipeline:
            for _index, frame, _timestamp in iter_capture(
                cap, max_frames=0, max_seconds=0, fps=info.fps
            ):
                result = pipeline.process(frame)
                for event in result.events:
                    line = {"event": event.describe(), "at": event.raised_at}
                    print(json.dumps(line) if args.json else f"ALERT  {event.describe()}")
                now = time.monotonic()
                if next_report is not None and now >= next_report:
                    if args.json:
                        _emit(pipeline.stats())
                    else:
                        _print_report(pipeline.stats())
                    next_report = now + args.report_every
                if deadline is not None and now >= deadline:
                    break
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    import numpy as np

    config = config_from_args(args)
    frames = _bench_frames(args, config.imgsz)

    with EdgePipeline.from_config(config) as pipeline:
        pipeline.warmup(frames[0], rounds=args.warmup)
        pipeline.reset()
        started = time.perf_counter()
        for i in range(args.frames):
            pipeline.process(frames[i % len(frames)])
        wall_s = time.perf_counter() - started
        report = {
            "backend": pipeline.stats()["backend"],
            "frames": args.frames,
            "frame_shape": list(np.shape(frames[0])),
            "wall_s": round(wall_s, 3),
            "throughput_fps": round(args.frames / wall_s, 2) if wall_s > 0 else 0.0,
            "latency": pipeline.latency.as_dict(),
        }

    if args.json:
        _emit(report)
    else:
        _print_report(report)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("serve needs `pip install -r app/requirements.txt`", file=sys.stderr)
        return 1
    uvicorn.run("app.api.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def _bench_frames(args: argparse.Namespace, imgsz: int) -> list:
    import numpy as np

    if not args.source:
        rng = np.random.default_rng(0)
        return [rng.integers(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)]

    from ppe.sources import classify_source, iter_capture, open_capture, read_image

    if classify_source(args.source) == "image":
        return [read_image(args.source)]
    cap = open_capture(args.source)
    try:
        frames = [frame for _i, frame, _t in iter_capture(cap, max_frames=30, max_seconds=0)]
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"No frames could be read from {args.source!r}")
    return frames


def _describe(result) -> str:
    if not result.workers:
        return f"no people ({len(result.detections)} boxes, {result.latency_ms:.1f} ms)"
    bad = [w.label for w in result.workers if w.violations]
    if not bad:
        return f"{len(result.workers)} worker(s), all compliant ({result.latency_ms:.1f} ms)"
    return f"{len(result.workers)} worker(s); " + "; ".join(bad)


def _print_report(report: dict, indent: int = 0) -> None:
    pad = " " * indent
    for key, value in report.items():
        if isinstance(value, dict):
            print(f"{pad}{key}:")
            _print_report(value, indent + 2)
        elif isinstance(value, list):
            if len(value) <= 8 and all(isinstance(v, (int, float, str)) for v in value):
                print(f"{pad}{key}: {', '.join(str(v) for v in value)}")
                continue
            print(f"{pad}{key}: {len(value)}")
            for item in value[:10]:
                print(f"{pad}  - {item}")
        else:
            print(f"{pad}{key}: {value}")


def _emit(payload) -> None:
    print(json.dumps(payload, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        return 130
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
