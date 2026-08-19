#!/usr/bin/env python3
"""Latency / FPS / memory: PyTorch vs ONNX Runtime at 640x640 and 960x540."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _common import REPO_ROOT, first_existing  # noqa: E402
from eval import resolve_weights  # noqa: E402

SIZES = ((640, 640), (960, 540))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--device", default=None, help="cuda / cpu / 0")
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "results" / "analysis" / "benchmark.json"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _rss_mb() -> float | None:
    try:
        import os

        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return None


def _vram_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except ImportError:
        return None
    return None


def _dummy(width: int, height: int):
    import numpy as np

    return np.zeros((height, width, 3), dtype="uint8")


def _time_ultralytics(model, image, warmup: int, iters: int, imgsz) -> dict[str, float]:
    for _ in range(warmup):
        model.predict(image, imgsz=imgsz, verbose=False)
    t0 = time.perf_counter()
    for _ in range(iters):
        model.predict(image, imgsz=imgsz, verbose=False)
    elapsed = time.perf_counter() - t0
    ms = (elapsed / iters) * 1000.0
    return {"latency_ms": ms, "fps": 1000.0 / ms if ms else 0.0}


def _time_ort(session, image, warmup: int, iters: int) -> dict[str, float]:
    import numpy as np

    inp = session.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    if len(shape) == 4:
        n, c, h, w = shape
        if h <= 0:
            h = image.shape[0]
        if w <= 0:
            w = image.shape[1]
        arr = np.zeros((n, c, h, w), dtype=np.float32)
    else:
        arr = np.zeros(shape, dtype=np.float32)
    feeds = {inp.name: arr}
    for _ in range(warmup):
        session.run(None, feeds)
    t0 = time.perf_counter()
    for _ in range(iters):
        session.run(None, feeds)
    elapsed = time.perf_counter() - t0
    ms = (elapsed / iters) * 1000.0
    return {"latency_ms": ms, "fps": 1000.0 / ms if ms else 0.0}


def main() -> int:
    args = _parse_args()
    weights = resolve_weights(args.weights)
    if weights is None:
        raise SystemExit("No weights found. Pass --weights path/to/best.pt")
    onnx_path = args.onnx
    if onnx_path is None:
        onnx_path = first_existing(weights.with_suffix(".onnx"), REPO_ROOT / "models" / "best.onnx")
    elif not onnx_path.is_absolute():
        onnx_path = REPO_ROOT / onnx_path
        if not onnx_path.exists():
            onnx_path = None

    plan = {
        "weights": str(weights),
        "onnx": str(onnx_path) if onnx_path else None,
        "sizes": [f"{w}x{h}" for w, h in SIZES],
        "warmup": args.warmup,
        "iters": args.iters,
        "backends": ["pytorch", "ultralytics_ort", "raw_ort"],
    }
    if args.dry_run:
        print("Dry-run benchmark plan:")
        print(json.dumps(plan, indent=2))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"dry_run": True, **plan}, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")
        return 0

    from ultralytics import YOLO

    rows: list[dict[str, Any]] = []
    pt_model = YOLO(str(weights))
    onnx_model = YOLO(str(onnx_path)) if onnx_path else None
    ort_session = None
    if onnx_path:
        try:
            import onnxruntime as ort

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ort_session = ort.InferenceSession(str(onnx_path), providers=providers)
        except Exception as exc:  # noqa: BLE001
            print(f"Raw ORT unavailable: {exc}", file=sys.stderr)

    for width, height in SIZES:
        image = _dummy(width, height)
        imgsz = [height, width]
        size_name = f"{width}x{height}"

        if hasattr(pt_model, "overrides") and args.device:
            pass
        rss0 = _rss_mb()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
        pt = _time_ultralytics(pt_model, image, args.warmup, args.iters, imgsz)
        rows.append(
            {
                "backend": "pytorch",
                "size": size_name,
                **pt,
                "rss_mb": _rss_mb(),
                "vram_mb": _vram_mb(),
                "rss_delta_mb": None if rss0 is None or _rss_mb() is None else _rss_mb() - rss0,
            }
        )

        if onnx_model is not None:
            ort_u = _time_ultralytics(onnx_model, image, args.warmup, args.iters, imgsz)
            rows.append(
                {
                    "backend": "ultralytics_ort",
                    "size": size_name,
                    **ort_u,
                    "rss_mb": _rss_mb(),
                    "vram_mb": _vram_mb(),
                }
            )

        if ort_session is not None:
            raw = _time_ort(ort_session, image, args.warmup, args.iters)
            rows.append(
                {
                    "backend": "raw_ort",
                    "size": size_name,
                    **raw,
                    "rss_mb": _rss_mb(),
                    "vram_mb": _vram_mb(),
                }
            )

    payload = {**plan, "results": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
