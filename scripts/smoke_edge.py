#!/usr/bin/env python3
"""Smoke the ONNX-first edge path (NPU-capable execution without requiring an NPU chip).

What this proves on any machine with ``onnxruntime``:

1. Torch is gated off by default (``PPE_ALLOW_TORCH`` unset).
2. A static-shape ONNX export of the inherited Construction weights loads via ORT.
3. ``ppe``-equivalent ``open_session`` + ``predict`` returns detections on a real image.
4. Hard NPU mode (``npu_only`` + ``allow_cpu_fallback=False``) fails loudly when no
   NPU EP is installed — so "NPU capable" deploys cannot silently become CPU-only.

On a Lunar Lake / OpenVINO box, set ``PPE_PROVIDERS=openvino,cpu`` and
``PPE_OPENVINO_DEVICE=AUTO:NPU,GPU`` before running; the same script will then
exercise the real NPU EP. Soft ``--npu-only`` (CPU fallback on) still runs here
when only CPU is installed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _common import REPO_ROOT  # noqa: E402

DEFAULT_WEIGHTS = REPO_ROOT / "baselines" / "snehilsanyal_yolov8n_css" / "models" / "best.pt"
DEFAULT_ONNX = REPO_ROOT / "models" / "best.onnx"
DEFAULT_SOURCE = (
    REPO_ROOT / "baselines" / "snehilsanyal_yolov8n_css" / "source_files" / "construction-safety.jpg"
)


def _ensure_onnx(weights: Path, onnx_path: Path, imgsz: int = 640) -> Path:
    if onnx_path.is_file():
        return onnx_path
    if not weights.is_file():
        raise SystemExit(f"Neither ONNX ({onnx_path}) nor source weights ({weights}) found")
    print(f"exporting {weights} -> {onnx_path}")
    from ultralytics import YOLO

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    exported = Path(
        model.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True, opset=17)
    )
    # Ultralytics writes next to the .pt; copy/rename into models/best.onnx.
    if exported.resolve() != onnx_path.resolve():
        onnx_path.write_bytes(exported.read_bytes())
    return onnx_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--require-npu",
        action="store_true",
        help="Hard NPU mode: fail if no NPU EP is available (PPE_ALLOW_CPU_FALLBACK=0)",
    )
    parser.add_argument("--skip-export", action="store_true", help="Require --onnx to already exist")
    args = parser.parse_args(argv)

    # Keep torch gated for this smoke unless the caller explicitly opted in.
    os.environ.setdefault("PPE_ALLOW_TORCH", "0")
    os.environ.setdefault("PPE_PREFER_ONNX", "1")

    from ppe.runtime import ExecutionPolicy, open_session, provider_status, resolve_providers
    from ppe.runtime.providers import has_npu_provider, require_npu_providers

    catalog = provider_status()
    npu_available = [r for r in catalog if r["tier"] == "npu" and r["available"]]
    print(json.dumps({"npu_available": npu_available, "resolved_default": resolve_providers(None)}, indent=2))

    hard_gate_checked = False
    if args.require_npu or os.environ.get("PPE_ALLOW_CPU_FALLBACK", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        # Prove hard mode: either an NPU EP is present, or we raise before a session opens.
        hard = ExecutionPolicy.from_env(npu_only=True, allow_cpu_fallback=False)
        resolved = resolve_providers(hard.providers or None, npu_only=True, allow_cpu_fallback=False)
        try:
            require_npu_providers(resolved, npu_only=True, allow_cpu_fallback=False)
        except RuntimeError as exc:
            print(f"HARD_NPU_GATE: {exc}")
            if not npu_available:
                # Expected on CPU-only CI / cloud VMs — refusal is correct; continue into the
                # soft ONNX predict below so the portable edge path is still exercised.
                print("HARD_NPU_GATE_OK (no NPU EP installed; refusal is correct)")
                hard_gate_checked = True
            else:
                raise
        else:
            if not has_npu_provider(resolved):
                raise SystemExit(f"expected an NPU provider, got {resolved}")
            hard_gate_checked = True
            print(f"HARD_NPU_GATE_OK (active NPU providers: {resolved})")

    if args.skip_export and not args.onnx.is_file():
        raise SystemExit(f"--skip-export set but {args.onnx} is missing")
    onnx_path = args.onnx if args.skip_export else _ensure_onnx(args.weights, args.onnx, args.imgsz)

    # Soft NPU preference: skip CUDA, allow CPU — this is the portable edge path.
    # (Hard mode was already verified above when --require-npu / PPE_ALLOW_CPU_FALLBACK=0.)
    if not hard_gate_checked and args.require_npu:
        raise SystemExit("internal error: --require-npu requested but hard gate was not checked")
    policy = ExecutionPolicy.from_env(npu_only=True, allow_cpu_fallback=True, prefer_onnx=True)
    session = open_session(onnx_path, policy=policy, repo_root=REPO_ROOT)
    info = session.info()
    print(json.dumps({"session": {
        "model": info["model_path"],
        "backend": info["backend"],
        "specialists": info.get("specialists"),
        "npu_only": info["policy"]["npu_only"],
        "allow_torch": info["policy"]["allow_torch"],
    }}, indent=2, default=str))

    if info["backend"].get("backend") != "onnxruntime":
        raise SystemExit(f"expected onnxruntime backend, got {info['backend']}")
    if info["policy"]["allow_torch"]:
        raise SystemExit("smoke expects Torch gated off")

    import cv2

    src = args.source if args.source.is_absolute() else REPO_ROOT / args.source
    image = cv2.imread(str(src))
    if image is None:
        raise SystemExit(f"failed to read {src}")
    dets, workers = session.predict_and_comply(image)
    summary = {
        "source": str(src),
        "detections": len(dets),
        "workers": len(workers),
        "classes": sorted({d.cls_name for d in dets}),
        "providers_active": info["backend"].get("providers_active"),
    }
    print(json.dumps(summary, indent=2))
    if not dets:
        raise SystemExit("edge predict returned zero detections on the smoke image — check the ONNX export")
    print("EDGE_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
