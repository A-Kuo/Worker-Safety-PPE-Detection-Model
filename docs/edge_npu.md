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

## Soft vs hard NPU mode

| Mode | Env | Behavior |
|---|---|---|
| Soft (default) | `PPE_NPU_ONLY=1` | Prefer NPU EPs; skip CUDA/TensorRT; **CPU still allowed** as fallback |
| Hard | `PPE_NPU_ONLY=1` **and** `PPE_ALLOW_CPU_FALLBACK=0` | Must resolve a real NPU EP (`openvino` / `qnn` / …) or **raise** before the session opens — no silent CPU |

CI runs `python scripts/smoke_edge.py --require-npu`: on CPU-only runners that proves the hard gate refuses; on a Lunar Lake box with `onnxruntime-openvino` it exercises the OpenVINO EP.

### Strict NPU mode (`PPE_STRICT_NPU=1` / `--strict-npu`)

Hard mode only proves an NPU EP was *requested*. ONNX Runtime always keeps an implicit CPU EP, assigns any op the NPU EP rejects to it, and — if an EP fails to load (e.g. a missing `openvino.dll`) — only logs a warning and runs on the CPU. Strict mode closes those gaps with one switch:

| Guard | What it does |
|---|---|
| No CPU EP | Implies `npu_only`, removes every CPU EP even if pinned (`PPE_PROVIDERS=openvino,cpu`) |
| No node fallback | Sets `session.disable_cpu_ep_fallback=1`: ORT refuses to build the session if any node would run on the CPU |
| EP must actually load | A requested provider missing from `session.get_providers()` is an error (a `RuntimeWarning` outside strict mode) |
| Device pinned | OpenVINO device must be `NPU` (default under strict). `GPU`, `CPU`, `AUTO:*`, `HETERO:*` are refused: they can land on the iGPU/CPU |
| Static shape | A dynamic input shape is refused (NPUs need static shapes) |

### Prove it: `ppe verify-npu`

```powershell
ppe verify-npu --weights models/best.onnx --providers openvino --out results/analysis/verify_npu.json
```

Runs a profiled inference and reads which provider executed each node. Exit code 0 only if nothing ran on the CPU **and** the OpenVINO device is pinned to the NPU. `--soft` diagnoses without enforcement and lists the exact CPU-resident ops. `providers_active` always contains the CPU EP (ORT appends it), so trust `by_provider` / `cpu_resident` / `npu_fraction`, not that list. The OpenVINO EP compiles the whole supported graph into one node, so a passing run shows `nodes_total: 1`.

`session.info()` also carries a runtime fingerprint (ORT distribution + version, OpenVINO version and NPU driver, Python/platform, model SHA-256 and opset) so a latency number can be reproduced or distrusted.

**Windows OpenVINO setup.** `onnxruntime-openvino` does not declare the `openvino` runtime as a dependency but the EP needs its DLLs (ORT 1.24 ↔ OpenVINO 2025.4). `pip install -e ".[edge-intel-npu]"` now includes it, and the runtime adds the DLL directory to the search path automatically.

Env: `PPE_STRICT_NPU`, `PPE_OPENVINO_CACHE_DIR` (compiled-model cache, skips NPU recompilation), `PPE_QNN_BACKEND` (QNN HTP library for Snapdragon).

## Local-only demo

The FastAPI + Streamlit demo defaults to **loopback execution** (`PPE_LOCAL_ONLY=1`): API host `127.0.0.1`, CORS limited to localhost Streamlit/API origins. Inference never leaves the machine — no cloud endpoints. Set `PPE_LOCAL_ONLY=0` / `PPE_API_HOST=0.0.0.0` only for intentional LAN demos.

Camera path: `GET /stream?timely=true` drains the capture buffer each tick so slow NPU/CPU ticks stay on the newest frame (see `app/media.py` `read_latest_frame`).

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
python scripts/export_onnx.py --weights archive/baselines/snehilsanyal_yolov8n_css/models/best.pt --imgsz 640 --out models/best.onnx

# 2) INT8 (dynamic; no calib folder required)
python scripts/quantize_onnx.py --model models/best.onnx
# optional static:
# python scripts/quantize_onnx.py --model models/best.onnx --calibration data/calib --static

# 3) Portable edge smoke (ORT + hard-NPU gate; works on CPU-only CI too)
python scripts/smoke_edge.py --require-npu

# 4) Bench (synthetic frame if --source omitted)
ppe bench --weights models/best.onnx --npu-only
ppe bench --weights models/best.int8.onnx --source archive/baselines/snehilsanyal_yolov8n_css/source_files/construction-safety.jpg --npu-only --providers openvino,cpu --openvino-device AUTO:NPU,GPU

# 5) Single-image predict + compliance
ppe predict --weights models/best.onnx --source archive/baselines/snehilsanyal_yolov8n_css/source_files/construction-safety.jpg --npu-only
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
| `PPE_CLASS_CONF` | Per-class confidence overrides, e.g. `no_vest=0.10,no_helmet=0.15`. The backend is asked for the lowest threshold in play and results are filtered per class, so a lower bar for one weak class doesn't loosen the rest. See `docs/experiments.md` ("Lifting `no_vest`") for measured recall vs false-positive tradeoffs. |
| `PPE_SPECIALIST` | Single specialist path (legacy). Prefer `PPE_SPECIALISTS` when attaching more than one. |
| `PPE_SPECIALISTS` | Comma-separated class-pair specialists (`.onnx` preferred for NPU, or `.pt` with `PPE_ALLOW_TORCH=1`). Each specialist replaces only its owned pair (`vest`/`no_vest` and/or `helmet`/`no_helmet`); pairs are disjoint so both can run on the same frame. Each costs an extra forward pass on the same ORT providers as the main model (OpenVINO NPU included). Class names come from ONNX metadata — export with `ppe export`. See `docs/experiments.md`. |
| `PPE_OPENVINO_DEVICE` | OpenVINO EP `device_type`, e.g. `AUTO:NPU,GPU` (only applies when `openvino` is an active provider) |
| `PPE_PREFER_NPU` | Demo/UI bias toward NPU providers + default OpenVINO `AUTO:NPU,GPU` (default true) |
| `PPE_LOCAL_ONLY` | Bind API to `127.0.0.1` and restrict CORS to localhost (default true). Set `0` only for intentional LAN demos. |

## Specialists on NPU

Accuracy lifts for vest / helmet ship as **extra ONNX graphs**, not a retrained 14-class head — so the NPU path stays ONNX-first:

```powershell
# After a specialist training run (vest and/or helmet):
ppe export --weights runs/train/vest_specialist/weights/best.pt --out models/vest_specialist.onnx
ppe export --weights runs/train/helmet_specialist/weights/best.pt --out models/helmet_specialist.onnx
# optional INT8 (re-check P/R after quantize):
python scripts/quantize_onnx.py --model models/vest_specialist.onnx
python scripts/quantize_onnx.py --model models/helmet_specialist.onnx

$env:PPE_PROVIDERS = "openvino,cpu"
$env:PPE_OPENVINO_DEVICE = "AUTO:NPU,GPU"
$env:PPE_SPECIALISTS = "models/vest_specialist.onnx,models/helmet_specialist.onnx"
$env:PPE_CLASS_CONF = "no_vest=0.10,no_helmet=0.15"
ppe predict --weights models/best.onnx --source archive/baselines/snehilsanyal_yolov8n_css/source_files/construction-safety.jpg --npu-only --providers openvino,cpu --openvino-device AUTO:NPU,GPU
```

`ppe predict` / `open_session` attach every specialist on the same provider list as the main model. If a `.pt` path is given while `PPE_PREFER_ONNX=1`, an adjacent `.onnx` / `.int8.onnx` sibling is preferred automatically.

## Vest model on the NPU (v0.1)

Accuracy with confidence intervals, the INT8 result and the latency table for the shipped vest model are in [`MODEL_CARD.md`](MODEL_CARD.md). Short version, same Lunar Lake machine: INT8 on the NPU in strict mode is the fastest configuration measured (p50 4.25-4.31 ms end to end over three runs); FP32 runs at 6.2-6.4 ms on the NPU and 5.1-5.5 ms on the iGPU. Score any model on your own labeled images with `ppe eval --weights <model.onnx> --images <dir>`.

**INT8 must keep the decode head in float** (`ppe quantize --static --qdq --per-channel --float-head --calibration <dir> --seed 0`). A fully quantized YOLO head scored AP50 0.0 on the CPU while the same file still worked on the NPU, so always evaluate a quantized model on every device you ship to. Dynamic INT8 (`ppe quantize` with no flags) is not recommended for this model.

## Measured latency of the runtime (baseline model)

`ppe bench` reports p50/p95/p99, a per-stage breakdown (`preprocess_ms`, `infer_ms`, `postprocess_ms`) and a reproducibility fingerprint. Raw results live in `results/bench/` (repo paths scrubbed).

**Setup:** Core Ultra 7 256V (NPU "Intel AI Boost" driver 1005540, Arc 140V iGPU), `onnxruntime-openvino` 1.24.1 + `openvino` 2025.4.1, FP32 ONNX, static 640x640, one real 1024x683 frame, 300 timed iterations after 20 warmup. **The model is the inherited Construction YOLOv8n baseline, not a vest model** — these numbers characterize the runtime, not vest accuracy. Single machine, not an isolated rig (normal Windows background load).

| device | p50 before | p50 after | p95 after | p99 after |
|---|---|---|---|---|
| NPU (strict) | 14.6 ms | **5.5–6.2 ms** (3 runs) | 6.8–12.8 | 7.5–31.9 |
| iGPU (OpenVINO GPU) | 13.7 | 4.3 | 8.7 | 19.9 |
| CPU (OpenVINO CPU) | 50.0 | 36.0 | 82.8 | 159.9 |
| CPU (ORT CPU EP) | 444 | 363 | 586 | 1678 |

"Before" is the same code without the preprocessing change. Take-aways:

- **Preprocessing was 38% of an NPU frame**, not the NMS (decode + NMS is ~0.6 ms). `Letterboxer` reuses buffers and writes a contiguous NCHW tensor: 4.7 → 1.1 ms with bit-identical output (`tests/test_letterbox.py` checks exact equality against the old implementation).
- **The inference stage also dropped (7.9 → ~3.8 ms)** because the old tensor was a non-contiguous view and ORT copied it inside `session.run`. A controlled A/B in one session, alternating inputs, measured the hidden copy at **+2.8 ms on the NPU and +3.2 ms on the iGPU**.
- Specialists now share one preprocessing pass per frame (was one per model). Not yet measured on real vest/helmet specialists — none are checked in.
- **On this model the iGPU is faster than the NPU at p50.** The NPU's case is offload and the strict no-CPU/no-iGPU guarantee, not raw speed; power draw has not been measured here, so no efficiency claim is made.
- Tail latency (p99) is noisy run to run on this machine. Do not quote p99 from these runs; re-measure on an idle, plugged-in system before publishing a tail figure.

Reproduce: `ppe bench --weights <model.onnx> --source <frame.jpg> --strict-npu --providers openvino --iterations 300 --warmup 20 --out results/bench/<name>.json`.

## Accuracy note

Use static QDQ quantization with `--float-head` (see "Vest model on the NPU" above) and re-check accuracy with `ppe eval` on each target device; the vest model's measured FP32-vs-INT8 numbers are in `MODEL_CARD.md`. Training accuracy work (Colab/Kaggle E0–E4) is separate from this runtime path.

## Intel Lunar Lake (AI Boost NPU + Arc iGPU)

Intended compute split: train on Kaggle/Colab T4 or P100 (see [`docs/compute.md`](compute.md)); run real-time edge inference locally on the Lunar Lake NPU/GPU via the OpenVINO EP.

**Package conflict — read before installing.** `onnxruntime-openvino` is a *separate* PyPI distribution from the plain `onnxruntime` package already in the `edge` extra. Both occupy the same `onnxruntime` import namespace and generally cannot be installed together in one environment. Use a dedicated environment for the NPU/OpenVINO path:

```powershell
python -m venv .venv-npu
.venv-npu\Scripts\Activate.ps1
python -m pip install -e ".[edge-intel-npu]"
```

**Heterogeneous device targeting.** `PPE_PROVIDERS=openvino,cpu` alone selects the OpenVINO EP but leaves its internal device choice at the EP's own default. To explicitly target the NPU with automatic fallback to the iGPU (and OpenVINO's own further fallback to CPU for unsupported ops), also set `PPE_OPENVINO_DEVICE`:

```powershell
$env:PPE_PROVIDERS = "openvino,cpu"
$env:PPE_OPENVINO_DEVICE = "AUTO:NPU,GPU"   # prioritize AI Boost NPU, fall back to Arc iGPU
ppe bench --weights models/best.onnx
```

`ppe bench`/`ppe predict` also accept `--openvino-device "AUTO:NPU,GPU"` directly instead of the env var. Other valid values: `NPU`, `GPU`, `CPU`, or an explicit heterogeneous list like `HETERO:NPU,GPU,CPU`. Confirm the active device via `session.info()`'s `openvino_device_type` field, or by checking OpenVINO's own device-selection log output.

**Static shapes.** `scripts/export_onnx.py --dynamic` already defaults to off (static 640×640), which the NPU requires — no extra flag needed for a plain ONNX export. For Ultralytics' native OpenVINO IR export instead of plain ONNX, use `--format openvino`:

```powershell
python scripts/export_onnx.py --weights <checkpoint>.pt --format openvino --imgsz 640
```

This writes a `*_openvino_model/` directory next to the source weights rather than a single `.onnx` file.

**Not yet done:** an OpenVINO-native (`nncf`) INT8 quantization path — the existing `scripts/quantize_onnx.py` uses generic `onnxruntime.quantization`, which already runs fine on the NPU via the OpenVINO EP. Revisit only if benchmarking shows a real accuracy/latency gap worth a second quantization stack.
