#!/usr/bin/env python3
"""Smoke the ONNX edge path without torch, third-party weights or an NPU chip.

By default the model is a generated few-kilobyte detector (``ppe.testing.make_tiny_detector``) with
the same interface as the real export, so this runs anywhere ``onnxruntime`` and ``onnx`` are
installed (``pip install ".[cpu,dev]"``) and proves:

1. Torch is gated off by default.
2. A static-shape ONNX model loads through ONNX Runtime and the runtime's preprocessing, decode
   and NMS return the expected boxes and classes.
3. With ``--require-npu``: hard NPU mode refuses when no NPU provider is installed, so an
   "NPU capable" deployment cannot silently become CPU-only. On a machine that has an NPU provider
   the same flag requires it to resolve.

Smoke a real model instead with ``--onnx model.onnx --source image.jpg`` (expects at least one
detection), or ``--weights best.pt`` to export first (needs the ``train`` extra).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _common import REPO_ROOT, ensure_repo_on_path  # noqa: E402

ensure_repo_on_path()


def check_hard_gate() -> None:
    """Hard NPU mode must either resolve an NPU provider or refuse before a session opens."""
    from ppe.runtime import ExecutionPolicy, provider_status, resolve_providers
    from ppe.runtime.providers import has_npu_provider, require_npu_providers

    npu_available = [r for r in provider_status() if r["tier"] == "npu" and r["available"]]
    hard = ExecutionPolicy.from_env(npu_only=True, allow_cpu_fallback=False)
    resolved = resolve_providers(hard.providers or None, npu_only=True, allow_cpu_fallback=False)
    try:
        require_npu_providers(resolved, npu_only=True, allow_cpu_fallback=False)
    except RuntimeError as exc:
        print(f"HARD_NPU_GATE: {exc}")
        if npu_available:
            raise
        print("HARD_NPU_GATE_OK (no NPU EP installed; refusal is correct)")
        return
    if not has_npu_provider(resolved):
        raise SystemExit(f"expected an NPU provider, got {resolved}")
    print(f"HARD_NPU_GATE_OK (active NPU providers: {resolved})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", type=Path, default=None, help="Real ONNX model to smoke (default: generated)")
    parser.add_argument("--weights", type=Path, default=None, help="Checkpoint to export to --onnx first")
    parser.add_argument("--source", type=Path, default=None, help="Image for a real model")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--require-npu", action="store_true", help="Also check the hard NPU gate")
    args = parser.parse_args(argv)

    # Keep torch gated for this smoke unless the caller explicitly opted in.
    os.environ.setdefault("PPE_ALLOW_TORCH", "0")
    os.environ.setdefault("PPE_PREFER_ONNX", "1")

    import numpy as np

    from ppe.runtime import ExecutionPolicy, open_session, resolve_providers

    print(json.dumps({"resolved_default": resolve_providers(None)}))
    if args.require_npu or os.environ.get("PPE_ALLOW_CPU_FALLBACK", "1").strip().lower() in {"0", "false", "no", "off"}:
        check_hard_gate()

    generated = args.onnx is None and args.weights is None
    with tempfile.TemporaryDirectory(prefix="ppe_smoke_") as tmp:
        if generated:
            from ppe.testing import DEFAULT_BOXES, DEFAULT_NAMES, make_tiny_detector

            imgsz = 64  # the generated boxes are defined in a 64x64 input
            onnx_path = make_tiny_detector(Path(tmp) / "tiny_detector.onnx", imgsz=imgsz)
            image = np.full((96, 128, 3), 114, dtype=np.uint8)  # non-square: exercises letterboxing
        else:
            imgsz = args.imgsz or 640
            onnx_path = args.onnx or Path(tmp) / "exported.onnx"
            if not onnx_path.is_absolute():
                onnx_path = REPO_ROOT / onnx_path
            if args.weights is not None and not onnx_path.is_file():
                from ppe.tools.export import export_model

                weights = args.weights if args.weights.is_absolute() else REPO_ROOT / args.weights
                print(f"exporting {weights} -> {onnx_path}")
                export_model(weights, onnx_path, imgsz=imgsz)
            if args.source is None:
                raise SystemExit("--source IMAGE is required when smoking a real model")
            import cv2

            src = args.source if args.source.is_absolute() else REPO_ROOT / args.source
            image = cv2.imread(str(src))
            if image is None:
                raise SystemExit(f"failed to read {src}")

        # Soft NPU preference: skip CUDA, allow CPU. This is the portable edge path.
        policy = ExecutionPolicy.from_env(npu_only=True, allow_cpu_fallback=True, prefer_onnx=True, imgsz=imgsz)
        session = open_session(onnx_path, policy=policy, repo_root=REPO_ROOT)
        info = session.info()
        if info["backend"].get("backend") != "onnxruntime":
            raise SystemExit(f"expected onnxruntime backend, got {info['backend'].get('backend')}")
        if info["policy"]["allow_torch"]:
            raise SystemExit("smoke expects Torch gated off")
        if "torch" in sys.modules or "ultralytics" in sys.modules:
            if generated:
                raise SystemExit("the generated-model smoke must not import torch / ultralytics")

        dets = session.predict(image)
        summary = {
            "model": "generated tiny detector" if generated else str(onnx_path),
            "detections": [(d.cls_name, round(d.conf, 3)) for d in dets],
            "providers_active": info["backend"].get("providers_active"),
            "static_input": info["backend"].get("static_input"),
            "stage_ms": {k: round(v, 3) for k, v in session.last_timings.items()},
        }
        print(json.dumps(summary, indent=2))
        if generated:
            want = sorted((DEFAULT_NAMES[c], round(conf, 3)) for c, conf, *_ in DEFAULT_BOXES)
            got = sorted(summary["detections"])
            if got != want:
                raise SystemExit(f"generated detector: expected {want}, got {got}")
            # Boxes were defined in 64x64 letterbox space on a 128x96 image: scale 0.5, pad (0, 8).
            first = max(dets, key=lambda d: d.conf)
            expect = (24.0, 8.0, 56.0, 40.0)
            if any(abs(a - b) > 0.5 for a, b in zip(first.xyxy, expect, strict=True)):
                raise SystemExit(f"letterbox/decode mismatch: expected {expect}, got {first.xyxy}")
        elif not dets:
            raise SystemExit("edge predict returned zero detections on the smoke image")
    print("EDGE_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
