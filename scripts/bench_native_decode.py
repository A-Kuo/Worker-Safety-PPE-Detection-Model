#!/usr/bin/env python3
"""Time the Python decode on the tensor used by ``native/ppe-core/examples/bench_decode.rs``.

Both programs build the same two (6, 8400) tensors (sparse and dense) from the same generator and print its sum and the
number of detections, so you can see they did the same work before comparing times.

    python scripts/bench_native_decode.py
    (cd native/ppe-core && cargo run --release --example bench_decode)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ppe.runtime.backends import _decode_yolo_onnx  # noqa: E402

MASK = (1 << 64) - 1


def tensor(sparse: bool, channels: int = 6, anchors: int = 8400) -> np.ndarray:
    state = 42
    out = np.zeros((channels, anchors), dtype=np.float32)

    def nxt() -> np.float32:
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & MASK
        return np.float32(state >> 40) / np.float32(1 << 24)

    for a in range(anchors):
        out[0, a] = nxt() * np.float32(640.0)
        out[1, a] = nxt() * np.float32(640.0)
        out[2, a] = np.float32(20.0) + nxt() * np.float32(200.0)
        out[3, a] = np.float32(20.0) + nxt() * np.float32(200.0)
        for c in range(4, channels):
            u = nxt()
            if sparse:
                for _ in range(9):  # u^512: a few dozen candidates, like a real frame
                    u = u * u
            else:
                u = u * u * u * u * u * u  # stress: thousands of candidates
            out[c, a] = u
    return out


def main() -> int:
    w0, h0, imgsz = 1280, 720, 640
    scale = min(imgsz / h0, imgsz / w0)
    meta = {"scale": scale, "pad": ((imgsz - round(w0 * scale)) // 2, (imgsz - round(h0 * scale)) // 2),
            "orig": (w0, h0), "imgsz": imgsz}
    names = {0: "0", 1: "1"}
    for label, sparse in (("sparse", True), ("dense ", False)):
        t = tensor(sparse)
        batch = [t[None, ...]]
        for _ in range(50):
            dets = _decode_yolo_onnx(batch, meta, conf=0.25, names=names)
        times = []
        for _ in range(2000):
            start = time.perf_counter()
            _decode_yolo_onnx(batch, meta, conf=0.25, names=names)
            times.append((time.perf_counter() - start) * 1e3)
        times.sort()
        print(
            f"python {label} 6x8400 conf 0.25: detections={len(dets):4d} p50={times[1000]:.4f} ms "
            f"p95={times[1900]:.4f} ms tensor_sum={float(t.astype(np.float64).sum()):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
