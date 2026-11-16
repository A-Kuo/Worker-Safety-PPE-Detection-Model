# Changelog

## 0.1.0 (beta) - unreleased

First public release. A vest / no-vest detector for on-device use, with a verifiable NPU-only mode.

**Status: beta.** The 95% precision / recall target is not met (vest P 0.897 / R 0.879, no_vest
P 0.648 / R 0.814 at confidence 0.30 on 249 / 43 test boxes). See `docs/MODEL_CARD.md`.

### Runtime
- ONNX Runtime backend with a provider registry (OpenVINO, QNN, Core ML, CUDA, DirectML, CPU).
- Strict NPU mode (`--strict-npu` / `PPE_STRICT_NPU`): no CPU provider, no CPU fallback, static input
  shapes, OpenVINO device pinned to the NPU, and an error if a requested provider fails to load.
- `ppe verify-npu`: profiles a run and reports the execution provider of every node.
- Reused-buffer preprocessing with bit-identical output (4.7 ms to 1.1 ms on the reference machine) and
  shared preprocessing across specialist models.
- Per-stage timings and a runtime fingerprint (versions, NPU driver, model SHA-256) in result files.
- Torch-free core install; `cpu` / `npu` runtime extras.

### Tools
- `ppe fetch-model`: verified download of released models (size and SHA-256 pinned in the package).
- `ppe window`: live window (PySide6) with a HUD that reports measured node placement.
- `ppe eval`: accuracy through the shipped runtime, with box counts and Wilson intervals.
- `ppe bench`: latency percentiles and stage breakdown.
- `ppe quantize --static --qdq --per-channel --float-head`: INT8 that stays within 0.02 AP50 of FP32 on
  NPU, iGPU and CPU. A fully quantized detection head scored AP50 0.0 on the CPU.
- `ppe export`, `ppe predict`, `ppe providers`.

### Models
- `vest_specialist.onnx` (FP32) and `vest_specialist.int8.onnx`, distributed as release assets.
- Measured on an Intel Core Ultra 7 256V: INT8 on the NPU 4.25 - 4.31 ms per frame (p50).

### Project
- Licence: AGPL-3.0-only. Third-party weights, plots, photographs and video without a stated licence
  were removed from the tree.
- CI on Ubuntu (Python 3.10 - 3.12) and Windows, including installing the built wheel in a fresh
  environment; draft-release and post-release workflows.

### Known limitations
- Frame-level detection only: no per-worker compliance without a person detector.
- Vests only. Evaluated on one Intel Lunar Lake machine; other NPUs are untested.
- Small `no_vest` test sample (43 boxes); no real-footage evaluation.
