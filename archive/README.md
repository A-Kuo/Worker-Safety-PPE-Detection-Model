# archive/

Material kept for provenance and reference, **not** part of the vest-focused core
(`vest` / `no_vest` detector, NPU-only runtime). Nothing here is imported by `src/`, `app/`,
`tests/` or CI, except as optional *fallback weight paths* and the smoke-test model (see below).

Layout mirrors the repo root so a file can be restored with `git mv archive/<path> <path>`.

| Path | What it is |
|---|---|
| `baselines/snehilsanyal_yolov8n_css/` | Inherited third-party Construction YOLOv8n (10 classes), including its ONNX export. Not trained here — see `ATTRIBUTION.md`. |
| `docs/baseline.md`, `docs/baseline_hf.md` | Audits of the Construction and Hexmon baselines. |
| `scripts/eval_baseline.py`, `eval_cross_domain.py` | Construction / Hard Hat Universe evaluation. |
| `scripts/fetch_checkpoint.py`, `reorder_checkpoint_classes.py`, `verify_reorder.py` | Hexmon checkpoint fetch, class-order fix and verification. |
| `scripts/finetune.py`, `configs/finetune/`, `configs/data/gap_finetune.yaml` | Gap fine-tune approach, superseded by the vest specialist. |
| `configs/data/construction.yaml`, `hardhat_eval.yaml` | Eval-only data configs for the archived eval scripts. |
| `configs/models/registry.yaml`, `models/pretrained/manifest.json` | Checkpoint candidate ledger and fetch manifest. |

Archived scripts import `_common.py` / `eval.py` from `scripts/` and resolve the repo root relative
to it, so they are **reference copies**: restore them to `scripts/` before running.

## Still in place on purpose

- `scripts/*` that `notebooks/train_colab_kaggle.ipynb` calls by path (Kaggle pulls scripts from
  GitHub, so these can be moved later together with a notebook update), plus `export_onnx.py` /
  `quantize_onnx.py` (used by `ppe export` / `ppe quantize`).
- The 14-class configs (`configs/train/e0_n` … `helmet_specialist`) and `scripts/train.py`
  `EXPERIMENTS`: `vest_specialist` fine-tunes from the E0 checkpoint.
- `app/` (Streamlit/FastAPI): replaced by the native NPU window in a later step.

## Temporary dependencies on `archive/baselines`

`scripts/smoke_edge.py` (run by CI) exports `archive/baselines/.../best.pt` and uses its sample image,
and the runtime / app default-weight lookups list it as a last fallback. The NPU-only runtime step
replaces these with a vest-specific model and removes the fallback.
