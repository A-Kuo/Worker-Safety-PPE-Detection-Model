<div align="center">

# 🦺 Unified Multi-Domain PPE Compliance Detection

**End-to-end industrial PPE detection & compliance analysis, built on YOLOv8 and multiple Roboflow Universe datasets.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)
[![Ultralytics YOLOv8](https://img.shields.io/badge/model-YOLOv8-purple.svg)](https://github.com/ultralytics/ultralytics)
[![Status: active development](https://img.shields.io/badge/status-active%20development-yellow.svg)](docs/PROJECT_PLAN.md)
[![Tests: 38 passing](https://img.shields.io/badge/tests-38%20passing-brightgreen.svg)](tests/)

<!--
PLACEHOLDER: hero demo GIF/image goes here.
Suggested: a short screen-capture of the FastAPI demo UI (once built, Milestone
M5) running detection on a webcam/video feed, boxes drawn, with a compliance
caption like "Worker 3 — missing helmet and vest" overlaid.
Save it to docs/assets/demo/hero_demo.gif and this line will render it:
-->
<img src="docs/assets/demo/hero_demo.gif" alt="PLACEHOLDER: replace with a demo GIF of live detection + compliance captions once Milestone M5 (deployment) is built" width="720" />

*This project is a personal portfolio demonstration (not a commercial product) —
see [Motivation](#-motivation) and [Licensing](#-licensing--attribution) below.*

</div>

---

## 📋 Table of contents

- [Overview](#-overview)
- [Motivation](#-motivation)
- [Current status](#-current-status--honest-progress-tracker)
- [Quickstart](#-quickstart)
- [Data](#-data)
- [Model architecture](#-model-architecture)
- [Training & experiments](#-training--experiments)
- [Evaluation & mathematical analysis](#-evaluation--mathematical-analysis)
- [Deployment](#-deployment-planned--milestone-m5)
- [Optional VLM layer](#-optional-vlm-layer-planned--milestone-m6)
- [Repository structure](#-repository-structure)
- [Future work](#-future-work)
- [Licensing & attribution](#-licensing--attribution)

---

## 🔎 Overview

This project detects Personal Protective Equipment (PPE) compliance in
industrial/construction imagery — helmets, vests, goggles, gloves, boots, and
masks, plus their **negative** counterparts (`no_helmet`, `no_vest`, etc.) so
the system can flag *missing* safety equipment, not just detect what's present.
It's built on **YOLOv8**, trained on a **unified schema merged from multiple
Roboflow Universe datasets** (rather than a single dataset), with an emphasis
on mathematical rigor — calibration analysis, safety-critical threshold
tuning, and cross-domain robustness evaluation — on top of the usual mAP
numbers.

<!--
PLACEHOLDER: a labeled sample detection image goes here — a real inference
output from this project's model (once trained) with bounding boxes and class
labels visible, ideally showing at least one "no_*" negative detection.
Save it to docs/assets/demo/sample_detection.jpg
-->
<img src="docs/assets/demo/sample_detection.jpg" alt="PLACEHOLDER: replace with a real annotated detection output from this project's trained model" width="600" />

**What makes this different from "YOLOv8 on one Roboflow dataset":**

| | |
|---|---|
| 🗂️ **Multi-source data** | Merges 4 dataset sources into one unified 17-class taxonomy — not just the original single Construction Site Safety dataset |
| 🧮 **Real math, not just mAP** | Calibration (ECE, Brier score) and minimum-recall-constrained threshold optimization for safety-critical classes, implemented as tested code in `src/evaluation/` |
| 🔬 **Reproducibility-first** | Every merge/label decision is a documented, unit-tested function (`src/data/label_schema.py`) — not a one-off notebook cell |
| 🌐 **Cross-domain aware** | Explicitly evaluates how performance shifts across the construction-site / lab / general-workplace domains the source datasets come from |
| 🚀 **Deployment-oriented** | Plans a FastAPI service + ONNX/INT8 export + edge latency benchmarks, not just a Jupyter notebook |

---

## 💡 Motivation

> 4,764 workers died on the job in 2020 (3.4 per 100,000 full-time equivalent
> workers). Workers in transportation/material-moving and construction/
> extraction occupations accounted for nearly half of all fatal occupational
> injuries (47.4%).
>
> — *Occupational Safety and Health Administration (US Department of Labor)*

Manual PPE compliance checks don't scale across large sites and shifts.
Automated detection — flagging *specifically what's missing* per worker,
not just "PPE present/absent" — is a natural computer-vision application with
a real safety case, and a good testbed for demonstrating rigor around
class-imbalance handling, calibration, and cross-domain generalization:
exactly the kind of applied-CV problem relevant to industrial safety
monitoring, defect inspection, and multi-modal scene understanding more
broadly.

---

## 📊 Current status — honest progress tracker

This README states real, measured facts only where they're backed by
something committed to this repo. Anything else is explicitly marked as
planned. Full detail and rationale for every item below: **[`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md)**.

| Milestone | Status | Detail |
|---|---|---|
| **M0** — Repo scaffold & experiment infra | ✅ Done | This repo's structure, `requirements.txt`, `.gitignore` |
| **M1** — Fork & reproduce baseline | 🟡 Half done | Documented the upstream repo's real training results ([`docs/BASELINE_METRICS.md`](docs/BASELINE_METRICS.md): mAP@0.5 = 0.810). Independent retraining is ⛔ **blocked** — needs GPU compute (see below) |
| **M2** — Expand & normalize datasets | 🟡 In progress | Merge pipeline **built, unit-tested, and smoke-tested against real data** ([`docs/DATA_DISTRIBUTION.md`](docs/DATA_DISTRIBUTION.md)). 2 of 4 sources still ⛔ **blocked on a Roboflow API key** |
| **M3** — Model experiments & math rigor | 🟡 In progress | Calibration/threshold-optimization **code implemented and unit-tested** (`src/evaluation/`, 38 tests passing) — not yet run against a real trained model (needs GPU, see M1) |
| **M4** — Cross-domain robustness | 📋 Planned | Blocked on M2/M3 |
| **M5** — Deployment & edge orientation | 📋 Planned | Blocked on M3 producing a model worth deploying |
| **M6** — Optional VLM layer | 📋 Planned (stretch) | Explicitly deferred until M1–M5 are solid |

**The single biggest unblock available right now:** a free Roboflow API key
(2-minute signup) plus running **[`notebooks/train_on_kaggle_or_colab.ipynb`](notebooks/train_on_kaggle_or_colab.ipynb)**
on Kaggle or Colab (this dev environment has no GPU). That one notebook run
unblocks M1's retraining, all of M2, and gives M3's evaluation code its first
real data to run against.

---

## 🚀 Quickstart

```bash
git clone https://github.com/A-Kuo/Worker-Safety-PPE-Detection-Model.git
cd Worker-Safety-PPE-Detection-Model
pip install -r requirements.txt
python -m pytest tests/   # 38 tests covering label remapping, dedup, and all eval math
```

To actually train/evaluate a model (requires a GPU and a free Roboflow API
key — this repo's own dev environment has neither):

1. Open [`notebooks/train_on_kaggle_or_colab.ipynb`](notebooks/train_on_kaggle_or_colab.ipynb) in Kaggle or Google Colab.
2. Enable a GPU accelerator.
3. Follow the notebook top-to-bottom — it exports all 4 dataset sources,
   merges them via `scripts/build_unified_dataset.py`, trains YOLOv8n/s, and
   runs this repo's own calibration/threshold-optimization code against the
   result.

---

## 🗂️ Data

Four sources, merged into one unified 17-class YOLO-format schema
(`configs/data_unified.yaml`, generated for real by `scripts/build_unified_dataset.py`
into `data/unified/data.yaml`). Full detail, exact confirmed class lists,
sizes, licenses, and the label-mapping rationale for every source:
**[`docs/DATASET_NOTES.md`](docs/DATASET_NOTES.md)**.

| Source | Images | Status | Notes |
|---|---|---|---|
| [Construction Site Safety](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety) | 2,801 | 🟡 Confirmed, export blocked | The original baseline dataset (10 classes, CC BY 4.0) |
| [PPE — Combined Model](https://universe.roboflow.com/roboflow-universe-projects/personal-protective-equipment-combined-model) | 44,002 | 🟡 Confirmed, export blocked | Adds `goggles`/`gloves`; **confirmed image-lineage overlap** with the base dataset — dedup is mandatory, not optional, when merging these two |
| [Hard Hat Universe](https://universe.roboflow.com/universe-datasets/hard-hat-universe-0dy7t) | 7,034 | 🟡 Confirmed, export blocked | 6 classes incl. explicit `head` (bare-head negative) and `hi-viz` variants |
| [Ultralytics Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe/) | 1,416 | ✅ Downloaded & merged | Added this session — no API key needed, already close to the target taxonomy. AGPL-3.0 licensed |

**Unified taxonomy** (`src/data/label_schema.py` is the source of truth):

```
Core PPE (positive):  helmet, vest, goggles, gloves, boots, mask
Negatives (missing):  no_helmet, no_vest, no_goggles, no_gloves, no_boots, no_mask
Scene / context:      person, machinery, vehicle, cone, ladder
```

**Real finding from the one merge run we've actually executed:** the dedup
step caught **57 of 122 duplicate clusters spanning more than one of
train/val/test** within a single official dataset — likely consecutive video
frames scattered across splits by whatever upstream split logic was used.
Full write-up: [`docs/DATA_DISTRIBUTION.md`](docs/DATA_DISTRIBUTION.md).

<!--
PLACEHOLDER: a class-distribution bar chart goes here, once all 4 sources are
merged (currently only 1 of 4 is reflected in docs/DATA_DISTRIBUTION.md).
Save it to docs/assets/demo/class_distribution.png
-->
<img src="docs/assets/demo/class_distribution.png" alt="PLACEHOLDER: replace with a full per-class distribution chart once all 4 sources are merged" width="600" />

---

## 🧠 Model architecture

- **Baseline**: YOLOv8n, matching the original upstream author's setup for a
  fair comparison ([`docs/BASELINE_METRICS.md`](docs/BASELINE_METRICS.md)).
- **Planned comparison**: YOLOv8s (accuracy/latency trade-off), standard vs.
  focal loss (class-imbalance handling), and an "industrial camera"
  augmentation preset (brightness/blur/occlusion/rotation) — see
  [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) M3 for the full experiment grid
  and why it's intentionally capped rather than exhaustive.
- **Stretch**: a CLIP/OpenCLIP layer for compliance captioning + retrieval
  (Milestone M6) — not started, deliberately deferred.

---

## 🧪 Training & experiments

Training happens on Kaggle/Colab (free GPU tier), not in this repo's own dev
environment — see **[`notebooks/train_on_kaggle_or_colab.ipynb`](notebooks/train_on_kaggle_or_colab.ipynb)**,
which implements the full pipeline: dataset export → merge → train → evaluate.

No training run has completed yet as of this README. Once one has, results
belong in a new `docs/EXPERIMENTS.md` (one row per run: backbone, loss,
augmentation config, global mAP@0.5 / mAP@0.5:0.95, per-class P/R/F1,
confusion matrix + PR curve images) — see
[`docs/BASELINE_METRICS.md`](docs/BASELINE_METRICS.md) for the format this
should follow once real numbers exist.

<!--
PLACEHOLDER: training curves (loss/mAP vs epoch) for this project's own runs,
once they exist. Save to docs/assets/demo/training_curves.png
-->
<img src="docs/assets/demo/training_curves.png" alt="PLACEHOLDER: replace with this project's own training curves once a run completes" width="600" />

---

## 📐 Evaluation & mathematical analysis

This is the part of the project built out furthest **ahead of having a
trained model** — the math is implemented and unit-tested now (38 tests
passing) so it's ready to run the moment real predictions exist. All of it is
pure Python, no `torch`/`ultralytics` required except for the one function
that actually calls a trained model (`src/evaluation/yolo_adapter.py`).

| Module | What it computes |
|---|---|
| [`src/evaluation/boxes.py`](src/evaluation/boxes.py) | IoU + greedy detection-to-ground-truth matching (COCO/YOLO-style TP/FP/FN assignment) |
| [`src/evaluation/metrics.py`](src/evaluation/metrics.py) | Per-class precision/recall/F1, micro/macro averages, confusion-matrix aggregation |
| [`src/evaluation/calibration.py`](src/evaluation/calibration.py) | Expected Calibration Error (ECE) and Brier score — is a 0.9-confidence detection right ~90% of the time? |
| [`src/evaluation/threshold_optimization.py`](src/evaluation/threshold_optimization.py) | Best-F1 threshold **vs.** a minimum-recall-constrained threshold for safety-critical classes (a missed `no_helmet` detection is worse than a false alarm) |

Formulas implemented (see [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) §4 for
the full derivation and rationale):

```
Precision = TP / (TP + FP)          Recall = TP / (TP + FN)          F1 = 2PR / (P + R)
ECE  = Σ_m (n_m/N) · |acc(m) − conf(m)|                  (per-bin calibration gap)
Brier = (1/N) · Σ (p_i − y_i)²                           (squared-error calibration score)
```

**Not yet run against real data** — no trained model exists yet in this repo
(this dev environment has no GPU). `notebooks/train_on_kaggle_or_colab.ipynb`
Section 6 runs this exact code against a real trained model's predictions;
results belong in a new `docs/CALIBRATION_ANALYSIS.md` once that happens.

<!--
PLACEHOLDER: reliability diagram (mean confidence vs. empirical accuracy)
and/or confusion matrix for this project's own trained model, once it exists.
Save to docs/assets/demo/reliability_diagram.png
-->
<img src="docs/assets/demo/reliability_diagram.png" alt="PLACEHOLDER: replace with this project's own reliability diagram once a model is trained and evaluated" width="500" />

---

## 🌐 Deployment (planned — Milestone M5)

Not started. Planned: a FastAPI service (batch + streaming inference
endpoints), a minimal web UI showing detections with compliance captions
("Worker 12 — missing helmet and vest"), and ONNX export + INT8 quantization
with latency/FPS/memory benchmarks at 640×640 and 960×540. Full spec:
[`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) M5.

<!--
PLACEHOLDER: screenshot of the web demo UI, once built.
Save to docs/assets/demo/webui_screenshot.png
-->
<img src="docs/assets/demo/webui_screenshot.png" alt="PLACEHOLDER: replace with a screenshot of the deployed web demo UI once Milestone M5 is built" width="700" />

---

## 🖼️ Optional VLM layer (planned — Milestone M6)

Not started, explicitly deferred until M1–M5 are solid. Planned: a
CLIP/OpenCLIP layer for compliance captioning ("two workers missing goggles
near scaffold") and text-based frame retrieval ("show all frames with missing
vests and boots"), evaluated with retrieval P@K on a small hand-labeled query
set. Full spec: [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) M6.

---

## 📁 Repository structure

```
.
├── README.md                       # you are here
├── LICENSE                         # MIT (this repo's own code only — see Licensing below)
├── requirements.txt
├── configs/
│   └── data_unified.yaml           # unified class list + dataset paths
├── data/                            # gitignored — populated by scripts/build_unified_dataset.py
├── docs/
│   ├── PROJECT_PLAN.md              # the full engineering plan, milestone-by-milestone
│   ├── DATASET_NOTES.md             # dataset facts + label mapping (start here for data questions)
│   ├── BASELINE_METRICS.md          # M1: real baseline metrics, documented from upstream
│   ├── DATA_DISTRIBUTION.md         # M2: real merge-pipeline output
│   └── assets/                      # committed reference plots/CSVs (small, curated)
├── notebooks/
│   └── train_on_kaggle_or_colab.ipynb   # run this to unblock everything GPU-related
├── scripts/
│   └── build_unified_dataset.py     # M2: reproducible dataset-merge pipeline
├── src/
│   ├── data/                        # label_schema.py + dedup.py (implemented, unit-tested)
│   ├── evaluation/                  # metrics/calibration/threshold-optimization (implemented, unit-tested)
│   ├── training/                    # not started
│   ├── serving/                     # not started (M5)
│   └── vlm/                         # not started (M6, stretch)
└── tests/                           # 38 tests, all passing
```

---

## 🔮 Future work

- Worker tracking + persistent ID association across frames.
- Integration into a robotics simulation or SCADA/MES-style monitoring system.
- Extending to additional modalities (thermal/IR) or environments beyond the
  current construction/lab/workplace domains.
- See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) §4 (M4–M6) for the
  concrete next milestones.

---

## ⚖️ Licensing & attribution

- **This repository's own code**: [MIT](LICENSE).
- **`ultralytics` (YOLOv8)**: AGPL-3.0. A non-issue for local/portfolio use;
  if this project's inference API is ever exposed as a live public service
  (not just demoed locally), AGPL's network-use clause applies — see
  [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md)'s risk register.
- **Datasets**: primarily CC BY 4.0 — see [`docs/DATASET_NOTES.md`](docs/DATASET_NOTES.md)
  for the exact license and citation per source. This project is a personal,
  non-commercial portfolio demonstration, not a redistribution of any dataset.
- **Ultralytics Construction-PPE dataset asset**: AGPL-3.0 (bundled under the
  `ultralytics/assets` release, same consideration as above).
