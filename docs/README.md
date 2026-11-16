# Documentation

| Document | Read it for |
|---|---|
| [MODEL_CARD.md](MODEL_CARD.md) | What the vest model is, measured accuracy with intervals, INT8, latency, limitations |
| [edge_npu.md](edge_npu.md) | Running on the NPU: strict mode, `ppe verify-npu`, getting the model, the live window, quantization |
| [experiments.md](experiments.md) | Training history, dataset caveats (partial labels, near-duplicates), why a specialist model |
| [compute.md](compute.md) | Where training runs (Kaggle / Colab) versus on-device inference |
| [RELEASING.md](RELEASING.md) | Cutting a release and attaching the model files |

Also at the repository root: `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`,
`ATTRIBUTION.md`, `NOTICE`, `LICENSE`.

Result files quoted by these documents live in `results/analysis/` (accuracy, placement) and
`results/bench/` (latency). `archive/` holds superseded tooling and notes.
