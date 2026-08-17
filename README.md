# Unified Multi-Domain PPE Compliance Detection

YOLOv8-based personal protective equipment (PPE) detection with a **unified 14-class schema**, person–PPE compliance association, and a local FastAPI + Streamlit demo. Construction Site Safety is documented as a **third-party baseline**; Hard Hat Universe is held out for domain evaluation.

| | |
|---|---|
| **Train schema** | Combined PPE v4 → 14 unified classes (`helmet` / `no_helmet`, …) |
| **Inherited baseline** | Construction YOLOv8n — mAP50 **0.809**, mAP50-95 **0.507** ([docs/baseline.md](docs/baseline.md)) |
| **Demo** | [app/README.md](app/README.md) — FastAPI + Streamlit |
| **Credits** | [ATTRIBUTION.md](ATTRIBUTION.md) |

---

## 1. Overview

This repo turns construction/industrial PPE detection into a reproducible portfolio pipeline:

1. **Audit** an inherited Construction Site Safety YOLOv8n baseline (not claimed as our training).
2. **Normalize** labels onto Combined PPE’s 14-class vocabulary.
3. **Train / evaluate** via YAML configs and scripts (E0–E4 grid; metrics pending GPU + `ROBOFLOW_API_KEY`).
4. **Calibrate** and cross-check domains; associate PPE boxes to `person` for compliance strings.
5. **Ship** a local API + UI for image, video, and webcam review.

Core library: [`src/ppe/`](src/ppe/) (`schema`, `compliance`, `inference`). Prefer `from ppe...` after `pip install -e .` (or with `src/` on `PYTHONPATH`); `src.ppe` remains a fallback in scripts/app.

---

## 2. Motivation

Missed PPE on site is a leading industrial safety failure mode. OSHA and similar regimes emphasize hard hats, high-visibility vests, eye/face protection, and related gear. Computer vision can flag **missing** equipment (`no_helmet`, `no_vest`, …) in camera feeds—similar in spirit to Matroid-style visual inspection—without replacing human judgment.

This project focuses on:

- A **shared label schema** across datasets so metrics are comparable.
- **Recall-first** operating points on violation classes (`no_*`).
- An honest split between **inherited artifacts** and **original engineering**.

---

## 3. Data

| Dataset | Role | Notes |
|---|---|---|
| [Construction Site Safety v28](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/28) | Inherited baseline train/eval | 10 classes; split **2605 / 114 / 82**; val n=114 is too small for strong per-class claims |
| [Personal Protective Equipment Combined Model v4](https://universe.roboflow.com/roboflow-universe-projects/personal-protective-equipment-combined-model/dataset/4) | **Unified train set** | ~44k images, 14 classes, 70/20/10 |
| [Hard Hat Universe](https://universe.roboflow.com/universe-datasets/hard-hat-universe-0dy7t) | **Held-out** helmet-domain eval | ~7k; not mixed into training |

**Licenses:** Roboflow Universe sets used here are **CC BY 4.0** — see [ATTRIBUTION.md](ATTRIBUTION.md).

**Label normalization** (Combined → unified):

| Combined raw | Unified |
|---|---|
| Hardhat / NO-Hardhat | `helmet` / `no_helmet` |
| Safety Vest / NO-Safety Vest | `vest` / `no_vest` |
| Goggles / NO-Goggles | `goggles` / `no_goggles` |
| Gloves / NO-Gloves | `gloves` / `no_gloves` |
| Mask / NO-Mask | `mask` / `no_mask` |
| Person / Safety Cone | `person` / `cone` |
| Ladder / Fall-Detected | `ladder` / `fall_detected` |

Construction-only `machinery` / `vehicle` stay on the Construction baseline; they are **not** in the unified model. **Boots are out of scope** (Combined has no `boots` / `no_boots`).

**No Construction ↔ Combined merge.** Construction already clones imagery from Combined and other Universe sets; merging without perceptual hashing would leak train/eval. Protocol: remap separately; evaluate Construction on mapped shared classes only.

Configs: [`configs/data/`](configs/data/). Distribution notes: [`docs/data_distribution.md`](docs/data_distribution.md).

---

## 4. Model Architecture

- **Baseline / primary detector:** Ultralytics **YOLOv8n** (nano) — Construction inherited weights under [`baselines/snehilsanyal_yolov8n_css/`](baselines/snehilsanyal_yolov8n_css/); unified runs use the same family.
- **Variants in the experiment grid:** YOLOv8**s** (E1); optional **m** only if E1 and E4 finish early.
- **Compliance layer:** `src/ppe/compliance.py` associates PPE boxes to `person` via containment / IoU and emits strings like `Worker k — missing helmet, vest`.
- **Optional VLM:** not implemented this cycle (see §8).

External reference only (not our checkpoint): [Hexmon/vyra-yolo-ppe-detection](https://huggingface.co/Hexmon/vyra-yolo-ppe-detection) (YOLOv8m on Combined v4).

---

## 5. Training & Experiments

Scripts and YAML drive training; nothing is claimed as a finished Combined run until GPU jobs complete with downloaded data.

| | |
|---|---|
| Configs | [`configs/train/`](configs/train/) — `e0_n`, `e1_s`, `e2_focal`, `e3_augs`, `e4_full44k` |
| Entry point | `python scripts/train.py --exp e0_n` |
| Protocol | [`docs/experiments.md`](docs/experiments.md) |

**Grid (one factor at a time, fixed seed / split):**

| ID | Change | Data |
|---|---|---|
| E0 | YOLOv8n default | Stratified Combined **12k** subset |
| E1 | YOLOv8s | Same subset |
| E2 | `fl_gamma=1.5` (focal) | Same subset |
| E3 | Stronger augs (blur, brightness, crop) | Same subset |
| E4 | YOLOv8n **50e** on full **44k** | Confirmation / shipped detector |

**Metrics pending** until `ROBOFLOW_API_KEY` downloads succeed and GPU training runs finish. Do not compare Combined 14-class mAP to Construction 10-class mAP as a like-for-like win; use `SHARED_EVAL_CLASSES` for fair slices.

---

## 6. Evaluation & Mathematical Analysis

### Inherited Construction baseline (documented, not our train)

From [`docs/baseline.md`](docs/baseline.md) / inherited `results.csv` (epoch 99):

| Metric | Value |
|---|---|
| mAP@0.50 | **0.809** |
| mAP@0.50:0.95 | **0.507** |
| Precision | **0.900** |
| Recall | **0.731** |

Per-class Ultralytics val needs Construction images on disk (`python scripts/eval_baseline.py`). Confusion-matrix takeaways (inherited plot): `NO-*` classes leak into background — motivation for recall-first threshold sweeps.

### Scripts for rigor (after Combined weights exist)

| Script | Purpose |
|---|---|
| `scripts/eval.py` | In-domain Combined eval |
| `scripts/eval_cross_domain.py` | Combined / Construction (mapped) / HHU tables |
| `scripts/calibrate.py` | ECE, Brier, `no_*` confidence sweeps (target R ≥ 0.90) |
| `scripts/benchmark.py` + `export_onnx.py` | Latency / FPS / memory (PyTorch vs ONNX) |

---

## 7. Deployment

Local demo only — see **[app/README.md](app/README.md)** for weights env vars and endpoints.

```bash
# From repo root
python -m pip install -r requirements.txt
python -m pip install -r app/requirements.txt
# Optional editable install for `from ppe...`:
python -m pip install -e .

# API
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
# or: python -m app.api.main

# UI (second terminal)
streamlit run app/ui/streamlit_app.py
```

OpenAPI: http://127.0.0.1:8000/docs — `POST /predict/image`, `POST /predict/video`, `GET /stream` (MJPEG).

Default weights: `PPE_WEIGHTS`, else first existing of `baselines/snehilsanyal_yolov8n_css/models/best.pt` or `models/best.pt`.

---

## 8. Optional VLM Layer (future)

Not implemented. A later stretch would freeze **OpenCLIP** (or similar) over a labeled frame bank for retrieval (“frames with missing vests”) — Matroid-shaped multimodal search without claiming captioning SOTA. Prefer retrieval over generative theater until the detector path is solid.

---

## 9. Future Work

- Multi-object **tracking** (ByteTrack / BoT-SORT) for stable worker IDs across frames
- **SCADA / MES** hooks for plant alarms and shift dashboards
- **IR / thermal** cameras for low-light and outdoor night shifts
- Optional YOLOv8m (E5) and INT8 quantization appendix after E4 latency numbers exist

---

## 10. What I inherited vs what I built

| Inherited (third-party) | Built in this repo |
|---|---|
| Snehil Sanyal Construction YOLOv8n weights, plots, `results.csv`, sample media | `src/ppe/` schema, compliance, inference |
| Original Roboflow Construction notes / yaml layout | `scripts/` download → remap → subset → analyze → train → eval → calibrate → export → benchmark |
| Artifact dump moved under `baselines/snehilsanyal_yolov8n_css/` | `configs/data`, `configs/train`, docs, tests |
| | FastAPI + Streamlit demo under `app/` |
| | Honest attribution and portfolio README |

**Do not** present `baselines/.../models/best.pt` as a model trained here.

### Citations

- **Snehil Sanyal** — [Construction-Site-Safety-PPE-Detection](https://github.com/snehilsanyal/Construction-Site-Safety-PPE-Detection)
- **Roboflow Universe** datasets above — **CC BY 4.0**
- **Hexmon/vyra-yolo-ppe-detection** — external Combined v4 reference only

Full license table: [ATTRIBUTION.md](ATTRIBUTION.md).

---

## Repo map

```text
src/ppe/           # schema, compliance, inference
scripts/           # pipeline CLIs (+ run_pipeline.md)
configs/data/      # construction, combined, hardhat_eval
configs/train/     # E0–E4 experiment YAMLs
app/               # FastAPI + Streamlit
docs/              # baseline, experiments, data_distribution
baselines/         # inherited Snehil Construction artifacts
tests/             # schema + compliance unit tests
```

### End-to-end command sequence

See also [`scripts/run_pipeline.md`](scripts/run_pipeline.md).

```bash
# Requires ROBOFLOW_API_KEY for downloads; GPU recommended for train/eval
python scripts/download_datasets.py --execute
python scripts/remap_labels.py --source data/raw/combined --out data/processed/combined --mapping combined
python scripts/remap_labels.py --source data/raw/hardhat --out data/processed/hardhat --mapping hhu
python scripts/remap_labels.py --source data/raw/construction --out data/processed/construction --mapping construction
python scripts/make_subset.py --source data/raw/combined --out data/raw/combined_12k --n 12000 --seed 42
python scripts/analyze_distribution.py
python scripts/train.py --exp e0_n
python scripts/eval.py --weights runs/train/e0_n/weights/best.pt
python scripts/calibrate.py --weights runs/train/e0_n/weights/best.pt
python scripts/export_onnx.py --weights runs/train/e0_n/weights/best.pt
python scripts/benchmark.py --weights runs/train/e0_n/weights/best.pt
# Then launch app/ (see §7)
```

### Quick checks

```bash
python -c "from ppe.schema import UNIFIED_CLASS_NAMES; print(len(UNIFIED_CLASS_NAMES))"  # → 14
pytest tests/
```
