# Model card: vest specialist (v0.1, beta)

A 2-class detector for hi-vis vests: `vest` and `no_vest` (a torso visibly without a vest).
Every number below is produced by `scripts/summarize_evidence.py` from committed JSON files that carry
the model's SHA-256. If a number here and a JSON disagree, the JSON is right.

**Status: beta. The project's 95% precision / recall target is not met.** See "Accuracy".

## Files

| Variant | File | SHA-256 | Size |
|---|---|---|---|
| FP32 (default) | `vest_specialist.onnx` | `b641539a7c56036382e9418b3df7a3b3f107f62f2059b65f4d8356e77e1c96b4` | 12,239,302 bytes |
| INT8 (optional) | `vest_specialist.int8.onnx` | `ebffbee7b9bb662a620a2fd935d2726a5d7a2f4d4f8ffdaf4fa6916735db2004` | 3,498,888 bytes |

Both: input `images` `[1, 3, 640, 640]` float32 RGB 0..1 (static), output `[1, 6, 8400]`, opset 17,
class names embedded in the ONNX metadata. Weights are distributed as release assets, not in git.

## Intended use

- Frame-level detection of worn and missing hi-vis vests in construction / industrial imagery, on
  device (Intel NPU, iGPU or CPU through ONNX Runtime).
- A screening aid for a human reviewer. It is **not** a safety system and must not be the only control.

**Out of scope / not supported**

- **Per-worker compliance.** The model has no `person` class, so it reports vest and no-vest boxes per
  frame, not "worker 3 is missing a vest". `ppe predict` returns `counts` and `detections`; its
  per-worker `compliance` list is empty by design. Associating boxes to people needs a person detector
  (roadmap).
- Other PPE (helmets, goggles, gloves, boots), other vest colours or domains not in the training data,
  identification of individuals.

## Training

| | |
|---|---|
| Architecture | Ultralytics YOLOv8n, 2 classes, COCO-pretrained initialization |
| Run | Kaggle T4, 30 epochs, imgsz 640, Ultralytics 8.4.37 (`configs/train/vest_specialist.yaml`) |
| Data | 3,253 training images: the vest-annotated part of Roboflow "Personal Protective Equipment Combined Model" v4, plus Roboflow `novest/no-vest-detect` v1 with images near-identical to any test image removed |
| Why a specialist | Combined merges sources that each labeled only some classes; 84% of worker images carry no vest label, which suppresses `no_vest` in a 14-class model (`docs/experiments.md`) |
| Licences | Datasets CC BY 4.0 (`ATTRIBUTION.md`); weights are a YOLOv8 derivative, AGPL-3.0 |

## Accuracy

**Test set:** 145 images, deduplicated against train/valid by perceptual hash, containing **249 `vest`
boxes and 43 `no_vest` boxes**. Scored with `ppe eval`: the shipped runtime (this package's decode and
class-agnostic NMS at IoU 0.45), detections kept from confidence 0.001, a detection counts when IoU with
an unmatched label is at least 0.5. Intervals are Wilson 95%.

FP32 on the NPU in strict mode (`results/analysis/eval_vest_onnx_clean_npu.json`):

| Class | Boxes | AP50 | Confidence | Precision (95% CI) | Recall (95% CI) | False boxes / 100 images |
|---|---|---|---|---|---|---|
| vest | 249 | 0.916 | 0.30 | 0.897 (0.853–0.930) | 0.879 (0.833–0.914) | 17.2 |
| no_vest | 43 | 0.711 | 0.30 | 0.648 (0.515–0.762) | 0.814 (0.674–0.903) | 13.1 |
| no_vest | 43 | 0.711 | 0.15 | 0.569 (0.448–0.682) | 0.861 (0.727–0.934) | 19.3 |

What this does and does not show:

- **Neither class meets the 95% target.** For both classes, at confidence 0.30, the whole 95% interval
  lies below 0.95, so the shortfall is real and not sampling noise.
- **43 `no_vest` boxes is a small sample.** The recall interval is 23 points wide. Do not read 0.81 as
  a precise figure, and do not compare models on this class by second-decimal differences.
- The iGPU and CPU agree with the NPU to within one or two boxes at any threshold, and AP50 matches to
  0.004 (`eval_vest_onnx_clean_{igpu,cpu}.json`).
- AP50 uses 101-point interpolation (the Ultralytics convention); its ceiling is 0.995.

**Operating point.** Use confidence **0.30** as the default. For missed-vest screening where recall
matters more than false alarms, lower only that class: `PPE_CLASS_CONF="no_vest=0.15"` raises `no_vest`
recall from 0.81 to 0.86 at the cost of about 6 more false boxes per 100 images. The package default is
still 0.25; a packaged per-model default arrives with the model fetcher.

**Provenance** (`results/analysis/eval_vest_ultralytics_crosscheck.json`, Ultralytics validator, NMS
IoU 0.7, same 145 images): the Kaggle checkpoint reproduces locally (vest AP50 0.919, no_vest 0.756);
its ONNX export scores 0.922 / 0.738; the shipped runtime path scores 0.916 / 0.711. The runtime figure
is the lowest of the three for `no_vest` and is the one quoted, because it is what users run.

## INT8 variant

Static QDQ quantization, per-channel weights, 200 seeded calibration images from the training split,
**decode head kept in float**:

```
ppe quantize --model models/vest_specialist.onnx --out models/vest_specialist.int8.onnx \
  --calibration data/processed/vest_specialist/train/images \
  --static --qdq --per-channel --float-head --calibration-images 200 --seed 0
```

`--float-head` is required. With the head quantized, box coordinates (0..640) and class scores (0..1)
share one scale at the output: measured AP50 was **0.0 on the CPU** while the NPU still scored 0.907 /
0.696, i.e. the model silently worked on one device and not another.

| Device | vest AP50 (FP32 → INT8) | no_vest AP50 (FP32 → INT8) |
|---|---|---|
| NPU (strict) | 0.916 → 0.915 | 0.711 → 0.719 |
| iGPU | 0.916 → 0.908 | 0.711 → 0.718 |
| CPU | 0.920 → 0.905 | 0.711 → 0.721 |

All within the 0.02 gate, and strict NPU placement is verified for both files
(`results/analysis/verify_npu_vest_{fp32,int8}.json`). At confidence 0.30 on the CPU, INT8 `no_vest`
recall is 0.767 against 0.814 for FP32: two boxes of 43, inside the noise, but check your own operating
point before switching.

## Latency

Intel Core Ultra 7 256V (Lunar Lake; NPU "Intel AI Boost" driver 1005540, Arc 140V iGPU),
`onnxruntime-openvino` 1.24.1, OpenVINO 2025.4.1. One test frame, 300 timed iterations after 20 warmup,
end to end (preprocess + inference + decode/NMS). Ranges are min–max over runs
(`results/bench/vest_latency_summary.json`).

| Model | Device | Runs | p50 ms | p95 ms | Inference p50 ms |
|---|---|---|---|---|---|
| INT8 | NPU (strict) | 3 | 4.25–4.31 | 5.55–5.68 | 2.56–2.60 |
| FP32 | iGPU | 3 | 5.09–5.47 | 6.14–6.45 | 3.11–3.52 |
| INT8 | iGPU | 3 | 5.92–6.16 | 6.81–7.42 | 3.89–4.17 |
| FP32 | NPU (strict) | 3 | 6.23–6.41 | 7.41–9.04 | 4.20–4.25 |
| INT8 | CPU (OpenVINO) | 1 | 14.4 | 16.2 | 12.5 |
| FP32 | CPU (OpenVINO) | 1 | 26.4 | 31.1 | 24.5 |

- INT8 on the NPU is the fastest configuration measured. In FP32 the iGPU is faster than the NPU.
- Preprocessing (about 1.3 ms) and decode/NMS (about 0.3 ms) are the same in every row.
- One machine, not isolated; p99 varied from 6 to 15 ms between runs and is not quoted as a guarantee.
- Power draw was not measured: no efficiency claim is made.

## Known limitations

- Accuracy target unmet; `no_vest` precision is low (roughly one false alarm in three at confidence 0.30).
- Small `no_vest` test sample (43 boxes); real-footage and cross-site evaluation not done.
- Training data inherits partial labelling and a construction-site bias.
- No calibration study; confidence values are not probabilities.

## Roadmap to the accuracy target

1. Collect and annotate more `no_vest` examples; grow the clean test set to several hundred `no_vest`
   boxes so a 95% claim can be tested at all.
2. Retrain, then re-run `ppe eval` and `scripts/validate_accuracy.py` (strict mode).
3. Ship a person detector so vest status can be attached to individual workers.
