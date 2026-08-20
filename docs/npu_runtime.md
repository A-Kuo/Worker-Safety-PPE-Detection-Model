# NPU inference

Inference runs on an NPU through ONNX Runtime. Nothing else is a deployment
target: torch is a training and export dependency, and the CPU path exists so
you can develop on a laptop, not so a camera can quietly fall back to it.

The rule this page enforces is that a host asked for an accelerator and denied
one says so at startup. ONNX Runtime does the opposite by default.

## Three execution policies

| Policy | Providers tried | On a host with no NPU |
|---|---|---|
| **`npu`** | NPU providers only, in registry order | Raises `NpuUnavailable` before the first frame |
| **`npu-preferred`** | NPU providers, then CPU | Runs on CPU, and `/health` says so |
| **`cpu`** | CPU only | Runs on CPU |

`npu` is the default. Choose `npu-preferred` when a mixed fleet is acceptable
and you would rather have degraded service than none. Choose `cpu` only for
development, and know that you are choosing it.

## Start here, every time

Ask the host what it has before you deploy anything to it:

```bash
ppe devices
```

```text
onnxruntime 1.29.0 on linux
installed providers: AzureExecutionProvider, CPUExecutionProvider

  [no ] QNNExecutionProvider       Qualcomm Hexagon NPU (Snapdragon X, 8cx)
        pip install onnxruntime-qnn  (Windows on ARM; needs the QNN SDK)
  [no ] OpenVINOExecutionProvider  Intel AI Boost NPU (Core Ultra)
        pip install onnxruntime-openvino openvino
  ...

No NPU execution provider is installed, so --execution npu will refuse to start.
```

It exits `0` when an NPU provider is present and `1` when none is, so a deploy
script can gate on it:

```bash
ppe devices --json > devices.json || { echo "no NPU on this host"; exit 1; }
```

The service reports the same thing at `GET /devices`, and `GET /health` carries
`execution`, `provider`, and `npu_available`.

## Supported accelerators

| Provider | Vendor | Hardware | Static shapes | Install |
|---|---|---|---|---|
| `QNNExecutionProvider` | Qualcomm | Hexagon NPU (Snapdragon X, 8cx) | required | `pip install onnxruntime-qnn` |
| `OpenVINOExecutionProvider` | Intel | AI Boost NPU (Core Ultra) | required | `pip install onnxruntime-openvino openvino` |
| `VitisAIExecutionProvider` | AMD | XDNA NPU (Ryzen AI) | required | `pip install onnxruntime-vitisai` |
| `CoreMLExecutionProvider` | Apple | Neural Engine | no | ships in the macOS wheel |
| `CANNExecutionProvider` | Huawei | Ascend 310P, 910 | required | build with `--use_cann` |
| `NnapiExecutionProvider` | Android | vendor NPU via NNAPI | required | build with `--use_nnapi` |
| `RknpuExecutionProvider` | Rockchip | RKNN (RK3588) | required | build with `--use_rknpu` |
| `VSINPUExecutionProvider` | VeriSilicon | Vivante NPU | required | build with `--use_vsinpu` |
| `DmlExecutionProvider` | Microsoft | DirectML device | no | `pip install onnxruntime-directml` |

DirectML is registered as `mixed`, not `npu`. It dispatches to whatever device
Windows offers, which on a machine with a discrete GPU is the GPU. Strict mode
excludes it. Pin it deliberately if you know the host has no GPU:

```bash
ppe watch rtsp://cam.local/stream --provider DmlExecutionProvider
```

## Getting a model the accelerator will accept

An NPU will not take an arbitrary float32 graph. Two properties matter.

**Static input shapes.** These providers compile the graph ahead of time and
cannot do that against a dynamic dimension. Export with fixed dimensions:

```bash
python scripts/export_onnx.py --weights runs/train/e4_full44k/weights/best.pt --imgsz 640
```

The script verifies the export afterwards and exits non-zero if any axis came
out dynamic. `--dynamic` is available for CPU work and produces a model the
strict NPU path refuses to load, by design.

**INT8 weights and activations.** Hexagon, XDNA, and Ascend run integer
kernels. Static QDQ quantization needs calibration frames:

```bash
python scripts/quantize_onnx.py \
  --model models/best.onnx \
  --calibration data/calib \
  --activation-type uint8
```

Use frames from the cameras the model will actually run on. A random slice of
the training set calibrates activation ranges that do not occur on site, and
the accuracy loss shows up later as missed violations at dusk.

Then confirm the numbers moved the way you expected, because quantization can
cost real mAP:

```bash
python scripts/eval.py --weights models/best.int8.onnx
ppe bench --weights models/best.int8.onnx --json
```

## Configuration

| Setting | Env | Default | Effect |
|---|---|---|---|
| `execution` | `PPE_EXECUTION` | `npu` | Policy from the table above |
| `provider` | `PPE_PROVIDER` | none | Pin one provider by name |
| `provider_options` | `PPE_PROVIDER_OPTIONS` | per provider | `key=value,key=value` overrides |
| `backend` | `PPE_BACKEND` | `auto` | `auto` always resolves to `onnx` |
| `weights` | `PPE_WEIGHTS` | see below | Path to the `.onnx` model |

Weights resolve to `models/best.int8.onnx`, then `models/best.onnx`, then the
inherited `.pt`, which is listed last only so `/health` can explain that it is
the wrong format rather than reporting a missing file.

Per-provider defaults the registry fills in for you:

| Provider | Defaults |
|---|---|
| QNN | `backend_path` for the platform, `htp_performance_mode=burst` |
| OpenVINO | `device_type=NPU`, `precision=FP16` |
| CoreML | `MLComputeUnits=CPUAndNeuralEngine`, `ModelFormat=MLProgram` |
| CANN | `device_id=0` |

Override any of them:

```bash
ppe watch 0 --provider-option htp_performance_mode=sustained_high_performance
export PPE_PROVIDER_OPTIONS="device_type=NPU,precision=ACCURACY"
```

## The torch backend

`UltralyticsBackend` still exists, for diffing an export against the checkpoint
it came from. Reaching it takes two deliberate acts:

```bash
PPE_ALLOW_TORCH=1 ppe image frame.jpg --backend ultralytics --weights best.pt
```

`--backend auto` never selects it, so a `.pt` path on a device fails with export
instructions instead of pulling torch onto hardware that should not have it.

## What to tell people about limits

Be accurate here. Overstating any of it is how this turns into a bad afternoon
on site.

- **Provider binding is verified.** Under `npu` the session is checked after
  construction and raises `ProviderNotBound` if ONNX Runtime bound CPU instead.
  That check is exact.
- **Node-level partitioning is not.** ONNX Runtime can place *some* operators
  on the NPU and the rest on CPU. The check above confirms the provider is
  first, not that every node landed there. A model that binds correctly and
  still runs slowly is usually partially partitioned; profile before assuming
  the accelerator is broken.
- **NNAPI is a request.** The Android layer picks its own backend, so a
  successful bind does not prove the DSP or NPU got the work.
- **CI cannot test any of this.** No hosted runner has an NPU. The tests cover
  registry logic, policy resolution, shape validation, and the failure paths,
  all against a CPU-only ONNX Runtime. The vendor code paths are exercised only
  on real hardware.
- **Quantization changes accuracy.** INT8 is a real approximation. Re-run
  evaluation on the quantized model; do not carry the float32 mAP forward.

## Troubleshooting

| Symptom | Cause | Do this |
|---|---|---|
| `NpuUnavailable` at startup | No NPU provider in this build | `ppe devices`, install the named wheel, or pass `--execution cpu` |
| `ProviderNotBound` | Provider loaded but rejected the graph | Usually a float32 or dynamic-shape model. Quantize and re-export |
| `needs a static input shape` | Exported with `--dynamic` | Re-export without it |
| Binds fine, throughput unchanged | Graph partitioned across NPU and CPU | Profile the session; unsupported operators are running on CPU |
| `ppe devices` shows nothing on a Copilot+ PC | Wrong wheel | `onnxruntime` and `onnxruntime-qnn` conflict; install only one |
| OpenVINO picks the iGPU | `device_type` not applied | Pin `--provider-option device_type=NPU` |
| Ascend device not found in a container | Devices not mapped in | Map the `davinci` devices and the CANN toolkit into the container |
| `/health` says `not an ONNX model` | `PPE_WEIGHTS` points at a `.pt` | Run `scripts/export_onnx.py` first |
