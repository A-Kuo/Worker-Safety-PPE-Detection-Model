# Edge / NPU inference

Default runtime is **ONNX Runtime** with a vendor-agnostic execution-provider registry. Torch / Ultralytics stays available only behind an opt-in flag (`PPE_ALLOW_TORCH=1` or `--allow-torch`).

## Decisions

| Choice | Value |
|---|---|
| Provider strategy | Vendor-agnostic EP registry (`src/ppe/runtime/providers.py`) |
| Default backend | ONNX (`.onnx` / `.int8.onnx`) |
| Torch | Gated opt-in — not used for NPU-only deploys |
| Quantization | Dynamic INT8 by default; optional static with `--calibration` |
| Boots | Still future work (no class in Combined) |

## Provider registry

Keys (preference for auto-select: NPU → GPU → CPU):

| key | ORT name | tier |
|---|---|---|
| `qnn` | `QNNExecutionProvider` | npu |
| `openvino` | `OpenVINOExecutionProvider` | npu |
| `coreml` | `CoreMLExecutionProvider` | npu |
| `cann` | `CANNExecutionProvider` | npu (Huawei Ascend) |
| `tensorrt` | `TensorrtExecutionProvider` | gpu |
| `cuda` | `CUDAExecutionProvider` | gpu |
| `dml` | `DmlExecutionProvider` | gpu |
| `cpu` | `CPUExecutionProvider` | cpu |

List what this machine actually has:

```powershell
ppe providers
# or
python -m ppe.cli providers
```

Pin providers:

```powershell
$env:PPE_PROVIDERS = "openvino,cpu"
$env:PPE_NPU_ONLY = "1"   # skip CUDA/TensorRT; CPU soft-fallback still on unless PPE_ALLOW_CPU_FALLBACK=0
```

## Export → quantize → bench

Use real paths (Windows treats `<...>` as redirection). Until Combined E4 exists, use the inherited Construction weights:

```powershell
# 1) FP32 ONNX (static 640 — better for many NPUs than dynamic axes)
python scripts/export_onnx.py --weights baselines/snehilsanyal_yolov8n_css/models/best.pt --imgsz 640 --out models/best.onnx

# 2) INT8 (dynamic; no calib folder required)
python scripts/quantize_onnx.py --model models/best.onnx
# optional static:
# python scripts/quantize_onnx.py --model models/best.onnx --calibration data/calib --static

# 3) Bench (synthetic frame if --source omitted)
ppe bench --weights models/best.onnx
ppe bench --weights models/best.int8.onnx --source baselines/snehilsanyal_yolov8n_css/source_files/construction-safety.jpg

# 4) Single-image predict + compliance
ppe predict --weights models/best.onnx --source baselines/snehilsanyal_yolov8n_css/source_files/construction-safety.jpg
```

Install the console script once:

```powershell
python -m pip install -e ".[edge]"
```

## Env reference

| Variable | Meaning |
|---|---|
| `PPE_WEIGHTS` / `PPE_MODEL` | Model path |
| `PPE_PROVIDERS` | Comma list of registry keys |
| `PPE_NPU_ONLY` | Prefer NPU EPs only |
| `PPE_ALLOW_CPU_FALLBACK` | Append CPU (default true) |
| `PPE_ALLOW_TORCH` | Opt-in Ultralytics `.pt` backend |
| `PPE_PREFER_ONNX` | Resolve `.onnx` before `.pt` (default true) |
| `PPE_IMGSZ` / `PPE_CONF` | Inference size / confidence |

## Accuracy note

INT8 dynamic quantization is a **best-effort** edge appendix. Re-check vest / helmet metrics after quantize; use static calibration images when mAP drops too far. Training accuracy work (Colab/Kaggle E0–E4) is separate from this runtime path.
