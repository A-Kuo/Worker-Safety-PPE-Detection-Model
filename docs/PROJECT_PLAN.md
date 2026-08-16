# Unified Multi-Domain PPE Compliance Detection — Project Plan

Status: started as a **planning artifact**; as of 2026-08-16 this repo also has
real, unit-tested code for dataset merging (`src/data/`, M2) and evaluation
math (`src/evaluation/`, M3) — see the milestone table in `README.md` for an
up-to-date status snapshot. No training has run yet (this dev environment has
no GPU) — see `notebooks/train_on_kaggle_or_colab.ipynb` for how to unblock
that on Kaggle/Colab.
Owner: Austin Kuo. Prepared by: Cursor Cloud Agent, 2026-08-15.

This document is the engineering plan for refactoring `Worker-Safety-PPE-Detection-Model`
from a single-dataset YOLOv8 fork into a multi-domain, mathematically-rigorous PPE
compliance system, per the 6-step approach you supplied. It keeps your original structure
but adds feasibility checks, grounded dataset facts, a concrete unified label schema,
math definitions for every metric mentioned, a risk register, and a milestone breakdown
that is scoped by **technical complexity**, not calendar days (per Cursor Cloud Agent
convention — “by end of the week” is treated as a target for the MVP slice, not the
full scope below).

See also: [`docs/DATASET_NOTES.md`](./DATASET_NOTES.md) for the grounded (web-verified)
facts about each dataset and the full label-mapping table.

---

## 0. Reality check — what fits in "this week" vs. what doesn't

The 6-step plan you wrote is the right *shape* for a strong portfolio project, but taken
literally and at full rigor (3 datasets merged and deduplicated, 2 backbones × 2 loss
functions × 4 augmentation configs = ~16 training runs, calibration + threshold tuning
per class, cross-domain evaluation, optional LoRA adapters, ONNX INT8 export + benchmarking,
*and* a CLIP retrieval/captioning layer) is a multi-week research effort even for an
experienced applied ML engineer with a dedicated GPU. Concretely:

- Each full YOLOv8 training run on the merged dataset (thousands of images, 100–200
  epochs) takes on the order of 1–4 GPU-hours depending on backbone size and hardware.
  Sixteen+ runs is not a single-session task.
- Dataset merging is not just "concatenate folders" — the three sources use different
  annotation conventions (e.g., `head` vs `helmet` vs `Hardhat`, presence of explicit
  `NO-*` negatives vs. implicit negatives), different image resolutions, and near-certainly
  overlapping/duplicate source images (Roboflow Universe forks frequently reuse the same
  underlying photos), which requires a real dedup + relabeling pass before any training
  is trustworthy.
- The VLM/CLIP layer (Step 6) and LoRA domain adapters (Step 4, optional) are genuinely
  separate sub-projects with their own evaluation protocols (retrieval P@K, captioning
  quality) — they should not be attempted until Steps 1–3 produce a trustworthy detector.

**Recommendation:** treat this plan as the full roadmap, but commit to an **MVP slice**
first (bolded milestones below: M0–M3, single backbone, single loss ablation, one
cross-domain eval) as the "end of this week" deliverable, and treat M4–M6 (robustness/LoRA,
edge quantization polish, VLM layer) as explicit fast-follow / stretch goals that get their
own PRs once the MVP is validated. This keeps every commit honest about what's actually
been measured versus what's aspirational.

---

## 1. Current repository state (as of this plan)

This cloud workspace's copy of `A-Kuo/Worker-Safety-PPE-Detection-Model` currently
contains **only** `README.md` and `.gitattributes` — there is no local file tree for a
YOLOv8/PPE project checked into this repo yet. The "local file tree copied from another
developer" you referenced is not present in this environment, and no PDFs
(`AustinKuo_FrwrdDepEng_v26__USDOD_-5.pdf`, `AustinKuo_DataEngineerCV_vT_26.pdf`) are
attached either. This plan is therefore grounded in:

1. Your written 6-step outline (taken as the source of truth for intent/structure).
2. Independently verified, current facts about the referenced upstream repos and
   datasets (web research, see `DATASET_NOTES.md`), since the PDFs weren't accessible.

**Update:** you've confirmed the "existing local files" project is
`snehilsanyal/Construction-Site-Safety-PPE-Detection` — the same repo already identified
below by research, so no re-identification was needed. Since it's a small (103MB),
public repo, I cloned it directly (shallow, inspection-only, not committed to this repo)
and pulled the real `results/results.csv` and result plots into
[`docs/BASELINE_METRICS.md`](./BASELINE_METRICS.md) — Step 1's "document exact mAP,
precision/recall per class, and confusion matrices from the existing `results/` folder"
is now done with real numbers, not placeholders. You also confirmed
`Vinayakmane47/PPE_detection_YOLO` gets "less inspiration" and wasn't copied — M5 below
has been adjusted to treat it as a loose reference only, not a structural base.

**Still open:** this cloud agent VM has no GPU and no `torch`/`ultralytics` installed,
so the *retraining* half of Step 1 (independently verifying the numbers are
reproducible, not just documenting the upstream author's numbers) isn't feasible in this
environment — see `BASELINE_METRICS.md` §4 for the concrete split between what's
CPU-feasible now (running inference/validation once we have the actual images) versus
what needs a GPU (full retraining, recommended on Kaggle/Colab as the original author
did).

---

## 2. Baseline source project (Step 1 target)

Verified upstream: **`snehilsanyal/Construction-Site-Safety-PPE-Detection`**
(GitHub, MIT-style notebook repo; dataset mirrored on Kaggle by the same author).

- Dataset: **Construction Site Safety Image Dataset** (Roboflow Universe, project
  `roboflow-universe-projects/construction-site-safety`), license **CC BY 4.0**.
- Size/split: **2,801 images** → `train: 2605 / valid: 114 / test: 82` (matches the
  split you cited exactly).
- **10 classes**: `Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person,
  Safety Cone, Safety Vest, machinery, vehicle`.
- Baseline model: YOLOv8n trained ~100 epochs on Kaggle P100.
- Published reference numbers exist from at least one independent reproduction
  (`Venta02/ppe-detection-yolov8`) for sanity-checking: YOLOv8n → mAP@0.5 ≈ 0.733,
  mAP@0.5:0.95 ≈ 0.419, P ≈ 0.904, R ≈ 0.665, best-F1 confidence ≈ 0.51. Treat these as
  a *plausibility check*, not ground truth — you must reproduce your own numbers from
  your own training run and your own `results/` artifacts.

Deployment reference: **`Vinayakmane47/PPE_detection_YOLO`** (Flask app, same 10-class
label set, `app.py` + upload/webcam flow) is the best match for "PPE_detection_YOLO" in
your Step 5. **You've confirmed this repo's files aren't copied and it should get "less
inspiration"** — M5 now treats it purely as a loose reference for the general
upload/webcam flow shape, and designs the FastAPI service from requirements directly
rather than porting its code.

---

## 3. Unified label taxonomy (Step 2 target)

Full source-class → unified-class mapping tables (per dataset, with confidence/caveats)
live in [`docs/DATASET_NOTES.md`](./DATASET_NOTES.md#unified-label-mapping). Summary of
the target schema:

| Group | Classes |
|---|---|
| **Core PPE (positive)** | `helmet`, `vest`, `goggles`, `gloves`, `boots`, `mask` |
| **Negatives (missing PPE)** | `no_helmet`, `no_vest`, `no_goggles`, `no_gloves`, `no_boots`, `no_mask` |
| **Scene / context** | `person`, `machinery`, `vehicle`, `cone`, `ladder` |

Key normalization decisions (see DATASET_NOTES for full rationale):

- `Hardhat`/`hardhat`/`helmet` → `helmet`; `NO-Hardhat`/`head` (bare head near a person
  box, no helmet match) → `no_helmet`.
- `Safety Vest` → `vest`; `NO-Safety Vest` → `no_vest`; Hard-Hat-Universe's `hi-viz vest`
  maps to `vest` (high-vis is a sub-attribute, not a separate class, to avoid excessive
  taxonomy fragmentation — logged as an open question in the risk register).
  `hi-viz helmet` → `helmet`.
  We should also consider preserving `hi_vis` as a boolean attribute rather than folding it,
  if downstream compliance rules need to distinguish "any vest" from "high-visibility vest."
- `Goggles`/`Mask` (Combined Model) → `goggles`/`mask` directly.
  **No explicit `no_goggles`/`no_mask`/`no_gloves`/`no_boots` classes exist in any of the
  three source datasets** — these negatives must be *synthesized*: for every `person` box
  with no IoU-matched positive PPE box of that type, emit a derived negative label. This
  is a real data-engineering task (Section 5, Step 2) and a documented limitation of
  the source data, not a simple relabel.
- `Safety Cone` → `cone`; `machinery`/`vehicle` pass through unchanged; Hard-Hat-Universe's
  `ladder`-adjacent context (if present) → `ladder`.

---

## 4. Milestones

Each milestone lists Objective, Concrete tasks, Deliverables (acceptance criteria), and
Key risks. Milestones map 1:1 to your original Steps 1–6, split where a step has
independently-verifiable sub-deliverables.

### M0 — Repo scaffold & experiment infra *(prerequisite, not in your original steps)*

- Objective: stand up a reproducible project skeleton before any training happens, so
  every later experiment is a config-driven, logged, diffable run rather than a notebook
  one-off.
- Tasks: directory layout (Section 6), `requirements.txt` pinned to a known-working
  `ultralytics`/`torch` combo, `configs/data_unified.yaml` skeleton, dataset download
  scripts that hit the Roboflow API (needs `ROBOFLOW_API_KEY` — see Open Questions),
  experiment-tracking convention (plain CSV/JSON run manifests is enough; W&B/MLflow
  optional stretch), fixed seeds + a `configs/*.yaml` per experiment so results are
  reproducible and diffable in PRs.
- Deliverables: this plan committed; empty-but-structured repo; a documented "how to run
  an experiment" recipe.
- Risks: over-engineering the scaffold before you know what you need — kept deliberately
  minimal (no custom training framework, just Ultralytics CLI/API + thin wrapper scripts).

### **M1 — Fork & reproduce baseline (your Step 1) — documentation half done**

- Objective: exact reproduction of YOLOv8n on the original 2605/114/82 split, with
  numbers you generated yourself and can defend.
- Status: **split into two sub-tasks, one done, one blocked on GPU access:**
  1. ✅ **Done** — documented the upstream repo's own `results/` folder verbatim into
     [`docs/BASELINE_METRICS.md`](./BASELINE_METRICS.md): global mAP@0.5 (0.810),
     mAP@0.5:0.95 (0.507), per-class AP@0.5 for all 10 classes (from the PR curve),
     per-class confusion-matrix diagonal, peak F1 (0.81 @ conf 0.488), per-class
     instance counts (from the labels histogram), and copies of the actual plots under
     `docs/assets/baseline/`.
  2. ⛔ **Blocked here** — independently re-running training to verify these numbers
     are reproducible. This cloud agent VM has no GPU and no `torch`/`ultralytics`
     installed. Recommended path: run on Kaggle (matching the original P100 setup) or
     Colab, using the same `data.yaml`/hyperparameters documented in
     `BASELINE_METRICS.md`, then bring the resulting `results/` folder back into this
     repo for comparison against the numbers already documented.
  3. A cheaper intermediate step that *is* CPU-feasible here: once we have API
     credentials to pull the actual validation images (Kaggle or Roboflow API key —
     see Open Questions), run `yolo val model=best.pt data=data.yaml` against them.
     That's inference on 114 images, not training, and would close the "exact per-class
     P/R/F1 at a fixed threshold" gap noted in `BASELINE_METRICS.md` §3c without needing
     a GPU.
- Risks: Kaggle P100 vs. your actual compute (Colab T4/L4, local GPU, CPU-only) will
  shift wall-clock and possibly final metrics slightly — document actual hardware used,
  don't claim P100 numbers if you trained elsewhere. The independent reproduction found
  during research (`Venta02/ppe-detection-yolov8`, mAP@0.5 ≈ 0.733) already differs
  meaningfully from the upstream author's own logged 0.810 — expect our retrain to land
  somewhere in that range, not an exact match, and report it as such rather than forcing
  a narrative of exact reproduction.

### **M2 — Expand & normalize datasets (your Step 2) — pipeline built & smoke-tested**

- Objective: merge the base dataset with the two Roboflow Universe sources you linked
  into one YOLO-format dataset under the unified schema from Section 3.
- **Confirmed (2026-08-16): you approved using the two Universe links as-is** for Step 2
  and delegated resolving any label/source discrepancies. Resolves Open Question #2.
- Status:
  1. ✅ **Done** — real, tested merge pipeline: `src/data/label_schema.py` (unified
     taxonomy + per-source class maps, raises loudly on any unrecognized class instead
     of silently dropping it), `src/data/dedup.py` (perceptual-hash near-duplicate
     detection), `scripts/build_unified_dataset.py` (orchestrates remap + copy + dedup +
     unified `data.yaml` + distribution CSV). 12 unit tests in `tests/`, all passing.
  2. ✅ **Done** — added a **fourth source**, `ultralytics_construction_ppe` (Ultralytics'
     own official "Construction-PPE" demo dataset, 1,416 images), because unlike the two
     Roboflow-hosted datasets it's downloadable with no API key and already uses almost
     exactly your target taxonomy (`helmet`/`vest`/`goggles`/`boots`/`gloves` +
     `no_*` negatives). This is an *addition*, not a substitution — the two Roboflow
     links remain the primary named sources; this one fills the `boots`/`goggles`/
     `gloves` gap neither of them confirms yet, and let the pipeline be smoke-tested
     against real images today instead of staying purely theoretical. One genuine
     ambiguity resolved along the way: this dataset's class literally named `"none"` was
     empirically confirmed (by rendering its own boxes on its own images — see
     `DATASET_NOTES.md`) to be its no-vest negative, not a class to drop.
  3. ✅ **Done (partial)** — ran the pipeline for real against this source and produced
     `docs/DATA_DISTRIBUTION.md`, including a real, unplanned finding: the dedup step
     caught **57 duplicate clusters (out of 122) spanning more than one of
     train/val/test** within this single dataset — i.e. even a well-known official
     dataset likely has some train/test leakage from consecutive video frames. This is
     exactly the kind of risk the dedup step exists to catch, now demonstrated on real
     data rather than asserted as a hypothetical risk.
  4. ⛔ **Still blocked** — `personal-protective-equipment-combined-model` (44,002
     images) and `hard-hat-universe-0dy7t` (~7,000 images) cannot be exported without a
     Roboflow API key (confirmed: the `roboflow` Python package raises
     `ValueError: A valid API key must be provided` even for these public datasets — see
     Open Questions). Both are already wired into `label_schema.py` with
     `status="pending_export"` and best-available class mappings from public research
     (locked-in exact class lists this session: Hard Hat Universe v26 = 6 classes,
     7,034 images, 4912/1414/708 split; Combined Model v4 = 44,002 images, 70/20/10
     split) — re-verify against the real export and flip to `status="confirmed"` once
     the key is available, then re-run `scripts/build_unified_dataset.py` with no other
     code changes needed.
  5. ⛔ **Not started** — IoU-based negative-class synthesis for sources that have no
     explicit negative for a class (needed for whichever classes the Combined Model
     export turns out to lack); cross-source dedup specifically between
     `construction_site_safety` and `ppe_combined_model` (the confirmed-not-hypothetical
     overlap from `DATASET_NOTES.md`); stratified re-split by source+class.
- Risks (high — this is the hardest step, matching your own assessment, and the dedup
  finding above is direct evidence of it): synthesized negatives are heuristic and will
  contain noise; annotation-style mismatch (bbox tightness, occlusion handling) between
  datasets can look like "domain shift" when it's actually "labeling-convention shift."
- **New risk found this session**: the `ultralytics` Python package (needed for all
  training) and the Ultralytics Construction-PPE dataset asset are both **AGPL-3.0
  licensed**. For a personal/portfolio, non-commercial project this is not a blocker,
  but AGPL's §13 "remote network interaction" clause means if Step 5's inference API is
  ever exposed as a live public service (not just run locally/demoed), the obligation to
  make corresponding source available kicks in. Worth a one-line disclosure in the
  final README if/when there's a public demo URL; a non-issue for local/portfolio use.

### **M3 — Model experiments & mathematical baseline (your Step 3) — eval code built & tested, no model to run it on yet**

- Objective: establish YOLOv8n baseline on the *unified* dataset, then run a scoped
  ablation grid with full metric + calibration reporting.
- Status: ✅ **The math/eval code is implemented and unit-tested** in `src/evaluation/`
  (`boxes.py` IoU+matching, `metrics.py` per-class P/R/F1 + confusion matrix,
  `calibration.py` ECE + Brier score, `threshold_optimization.py` best-F1 vs.
  minimum-recall-constrained thresholds, `yolo_adapter.py` to plug real Ultralytics
  predictions into all of the above) — 26 additional unit tests beyond M2's 12, all
  passing, using synthetic data since no trained model exists yet. ⛔ **Not yet run
  against real data** - that requires a trained model, which requires GPU compute this
  environment doesn't have. `notebooks/train_on_kaggle_or_colab.ipynb` Section 6 runs
  this exact code against real predictions the moment you have a `best.pt` - no further
  code needed, just execution.
- Experiment grid (recommend cutting to what's realistic for the MVP week — see Section 0):
  - Backbones: YOLOv8n (baseline) → YOLOv8s (1 additional run for the MVP; YOLOv8m as
    stretch).
  - Loss: standard (Ultralytics default: CIoU + BCE + DFL) vs. focal-loss variant on the
    classification term to address class imbalance.
  - Augmentation: default Ultralytics augs vs. an "industrial camera" preset (brightness/
    contrast jitter, motion blur, synthetic occlusion, small-angle rotation) — one A/B is
    enough to demonstrate the method; don't need a full factorial grid for the MVP.
- Metrics per experiment (definitions, so results are unambiguous in the writeup):
  - **mAP@0.5** and **mAP@[.5:.95]**: standard COCO-style mean AP, averaged over IoU
    thresholds 0.5:0.05:0.95 for the second metric.
  - **Per-class Precision/Recall/F1**: `P = TP/(TP+FP)`, `R = TP/(TP+FN)`,
    `F1 = 2PR/(P+R)`, computed at the confidence threshold that maximizes F1 (report the
    threshold value, don't just report the metric).
  - **Confusion matrix & PR curves**: use Ultralytics' built-in `results/` outputs
    (mirrors the snehilsanyal repo's approach) — commit the images alongside the numbers.
  - **Focal loss**: `FL(p_t) = -α_t (1-p_t)^γ log(p_t)`; report the `α`/`γ` used and why
    (start from `α=0.25, γ=2.0`, the RetinaFocal defaults, then justify any deviation with
    the imbalance numbers from M2's distribution analysis).
  - **Calibration — ECE**: bin predicted confidences into `M` bins, compute
    `ECE = Σ_m (n_m/N) |acc(m) − conf(m)|` where `acc(m)` is the fraction of
    IoU-matched true positives in bin `m` and `conf(m)` is the mean predicted confidence
    in that bin.
  - **Calibration — Brier score**: `BS = (1/N) Σ (p_i − y_i)²` over all predicted boxes
    matched to a ground-truth presence/absence label at a fixed IoU threshold.
  - **Threshold optimization for safety-critical classes**: for `no_helmet`, `no_vest`,
    etc., pick the confidence threshold that satisfies a *minimum recall constraint*
    (e.g., R ≥ 0.95) and reports the resulting precision/FP rate at that operating point —
    frame this explicitly as "we accept more false positives to avoid missing an unsafe
    worker," which is the right safety-critical framing and worth stating outright in the
    README.
- Deliverables: `docs/EXPERIMENTS.md` — one row per run with all metrics above,
  confusion matrix + PR curve images, calibration plots (reliability diagrams), and the
  chosen operating threshold per safety-critical class.
- Risks: running a full factorial grid is the most likely place this project blows its
  time budget — cap it explicitly (Section 0) and log any skipped cells as "future work"
  rather than silently dropping them.

### M4 — Robustness & cross-domain analysis (your Step 4) — **stretch beyond MVP**

- Objective: train once on the merged/unified dataset, evaluate separately per source
  domain (construction-site, combined-model's environments, hard-hat-universe) to
  quantify generalization vs. specialization trade-offs.
- Tasks: hold out per-source test splits (already required by M2's stratified split);
  report the M3 metric suite per domain; discuss whether a single general model or
  per-domain fine-tunes/LoRA-style adapter heads perform better, with numbers, not just
  narrative.
- LoRA/domain-adapter head note: standard YOLOv8 doesn't ship LoRA support out of the box;
  this would mean either (a) freezing the backbone and fine-tuning only the detection head
  per domain (a cheap, well-understood approximation of "adapter" behavior for CNN
  detectors), or (b) a genuine LoRA injection into the backbone conv/linear layers, which
  is a non-trivial custom-implementation task. Recommend (a) unless you specifically want
  to demonstrate LoRA mechanics for the Matroid-alignment narrative.
- Deliverables: `docs/CROSS_DOMAIN_EVAL.md` with per-domain metric tables and a clear
  recommendation (general vs. specialized) backed by the numbers.

### M5 — Deployment & edge orientation (your Step 5)

- Objective: modernize `Vinayakmane47/PPE_detection_YOLO`'s Flask skeleton into a proper
  service, add a demo front-end, and produce real edge-latency numbers.
- Tasks: FastAPI service with `/infer/image`, `/infer/batch`, and a streaming endpoint
  (websocket or MJPEG) for webcam/RTSP; minimal HTML/JS or Streamlit front-end rendering
  boxes + a compliance sentence per detected person (e.g., "Worker 3 — missing helmet and
  vest") using the IoU-based person↔PPE association logic already built in M2; ONNX export
  (`model.export(format="onnx")`) and INT8 post-training quantization; latency/FPS/memory
  benchmarks at 640×640 and 960×540 on whatever hardware you actually have (state it
  plainly — CPU-only laptop numbers are still useful and honest, don't imply NPU/edge
  hardware you didn't test on).
- Deliverables: `src/serving/` FastAPI app + minimal UI; `docs/DEPLOYMENT_BENCHMARKS.md`
  with a latency/FPS/memory table (PyTorch fp32 vs. ONNX fp32 vs. ONNX INT8, at both
  resolutions).

### M6 — Optional VLM layer (your Step 6) — **explicit stretch goal, own PR**

- Objective: add a CLIP/OpenCLIP layer for compliance captioning and text-based frame
  retrieval, to demonstrate multimodal thinking beyond bounding boxes.
- Tasks: embed cropped detections + full frames with `open_clip`; template-based
  captioning driven by the same compliance logic as M5 ("two workers missing goggles near
  scaffold"); simple retrieval demo (`"missing vest and boots"` → ranked frame list) using
  cosine similarity over CLIP text/image embeddings.
- Metrics: retrieval precision@K on a small hand-labeled query set (you'll need to write
  ~10–20 test queries with known correct frames — there's no existing benchmark for this,
  so be upfront that it's a small qualitative eval, not a large-scale one.
- Deliverables: `src/vlm/` module + `docs/VLM_LAYER.md` with example prompts/outputs and
  retrieval P@K numbers on the small hand-labeled set.

---

## 5. Proposed repository structure

```
.
├── README.md                      # top-level overview, links into docs/
├── requirements.txt
├── configs/
│   └── data_unified.yaml          # unified class list + dataset paths
├── data/                           # gitignored; raw + processed datasets live here
├── docs/
│   ├── PROJECT_PLAN.md             # this file
│   ├── DATASET_NOTES.md            # grounded dataset facts + label mapping
│   ├── BASELINE_METRICS.md         # M1 output (done)
│   ├── DATA_DISTRIBUTION.md        # M2 output (in progress - 1 of 4 sources merged so far)
│   ├── EXPERIMENTS.md              # M3 output
│   ├── CROSS_DOMAIN_EVAL.md        # M4 output
│   ├── DEPLOYMENT_BENCHMARKS.md    # M5 output
│   ├── VLM_LAYER.md                # M6 output
│   └── assets/                     # committed reference plots/CSVs/configs (small, curated)
│       ├── baseline/                # M1: copied results/ artifacts from the upstream repo
│       └── m2_smoke_test/           # M2: real pipeline output from the one unblocked source
├── scripts/
│   └── build_unified_dataset.py    # M2: reproducible merge pipeline (implemented + smoke-tested)
├── src/
│   ├── data/                       # M2: label_schema.py + dedup.py (implemented + unit-tested)
│   ├── training/                   # training entrypoints + experiment configs (not started)
│   ├── evaluation/                 # metrics, calibration, threshold tuning (not started)
│   ├── serving/                    # FastAPI app + inference wrappers (not started)
│   └── vlm/                        # M6 CLIP/OpenCLIP layer (not started)
├── notebooks/                      # exploratory EDA, kept out of the critical path
└── tests/                          # test_label_schema.py, test_dedup.py (12 tests, passing)
```

## 6. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Full experiment grid (Step 3) exceeds available time/compute | High | Delays MVP | Cap grid per Section 0; log skipped cells as future work |
| Synthesized negative labels (`no_goggles`, etc.) are noisy | High | Undermines metric validity | Document heuristic explicitly; manually spot-check a sample; report as a named limitation |
| Cross-dataset image duplication causes train/test leakage | **High (confirmed, not hypothetical)** — the base dataset's own `README.dataset.txt` states its images were cloned from `personal-protective-equipment-combined-model` (link #1 in the outline) among other sources | Inflated/misleading metrics if unresolved | Perceptual-hash dedup between the base dataset and the Combined Model export specifically, before any merged splitting (M2); check for train/test-split conflicts, not just raw duplicates |
| Annotation-convention differences mistaken for domain shift | Medium | Wrong conclusions in M4 | Explicitly separate "labeling convention" vs. "visual domain" in writeup |
| ~~`"Safety Goggles"` / `"PPE-Additions-2314"` naming mismatch~~ | **Resolved** — you confirmed using the two Universe links as-is | — | — |
| Roboflow export requires an API key even for public datasets | **Confirmed** (the `roboflow` package raises `ValueError: A valid API key must be provided` with no key at all, not just for paid tiers) | Blocks M2's remaining two sources | Need `ROBOFLOW_API_KEY` in Cursor Dashboard secrets; pipeline code is ready, just needs credentials to run against real exports |
| LoRA-for-CNN-detector claim is technically unusual and could read as buzzword-chasing in a portfolio review | Low–Medium | Credibility | Either implement head-freezing/fine-tuning honestly labeled as "adapter-style," or skip and say so |
| VLM layer scope creep turns a detection project into an unfinished multimodal project | Medium | Dilutes core deliverable | Keep M6 as a strictly separate, clearly-labeled stretch PR after M1–M5 are solid |
| `ultralytics` package (and the Ultralytics Construction-PPE dataset asset added this session) are **AGPL-3.0** licensed | Low for a personal/portfolio project | AGPL's network-use clause (§13) would require publishing corresponding source if M5's API is ever exposed as a live public service, not just demoed locally | Non-issue for local/portfolio use; add a one-line license disclosure to the README if a public demo URL is ever stood up |
| Official third-party datasets can themselves have train/test leakage | **Confirmed, not hypothetical** — dedup found 57/122 duplicate clusters spanning >1 split within the Ultralytics Construction-PPE dataset alone (likely consecutive video frames), see `DATA_DISTRIBUTION.md` §3 | Optimistic metrics if trained/evaluated on the original splits | Re-split by duplicate cluster (never split a cluster across train/val/test) before trusting any metrics from this or any other merged source |

## 7. Open questions (need your input before M2 can start for real)

1. ~~Push or share the local file tree / fork URL.~~ **Resolved** — confirmed as
   `snehilsanyal/Construction-Site-Safety-PPE-Detection`; documented in
   `BASELINE_METRICS.md`.
2. ~~"Safety Goggles" / "PPE-Additions-2314" naming mismatch.~~ **Resolved (2026-08-16)**
   — you confirmed using `personal-protective-equipment-combined-model` and
   `hard-hat-universe-0dy7t` as the two additional sources, and delegated resolving any
   further label/source ambiguity. Locked in this session: exact confirmed class lists
   and sizes for both (see `DATASET_NOTES.md`), plus a fourth source
   (`ultralytics_construction_ppe`) added to make real progress while these two remain
   blocked on API access (item #3).
3. **Still open, now the main blocker.** Do you have (or can you provision) a Roboflow
   API key for programmatic export? Confirmed this session: the `roboflow` Python
   package outright refuses to talk to the API without one — even for these fully public
   datasets — so this isn't optional the way it might be for a paid-tier limit. Add it as
   `ROBOFLOW_API_KEY` via Cursor Dashboard → Cloud Agents → Secrets. A Kaggle API key
   (`KAGGLE_USERNAME`/`KAGGLE_KEY`) would separately unblock the base dataset's actual
   images (also not committed to its GitHub repo — see `BASELINE_METRICS.md` §4) for the
   CPU-feasible validation-only run described there.
4. What compute do you actually have available for training (Colab/Kaggle free tier, a
   local GPU)? **Partially answered by this session**: this cloud agent VM itself has no
   GPU and no ML libraries installed, so it cannot run the M1 retrain or the M3
   experiment grid — those need to happen on Kaggle/Colab (or another GPU-backed
   environment) regardless of what you tell me here. CPU-only work (inference/validation
   on already-trained checkpoints, data merging/remapping scripts, the FastAPI service)
   remains fully doable in this environment.
5. Can you confirm whether the two attached PDFs (`AustinKuo_FrwrdDepEng_v26...`,
   `AustinKuo_DataEngineerCV_vT_26...`) contain additional constraints (e.g., a specific
   target audience like Matroid) that should reshape prioritization — they weren't
   accessible in this environment, so this plan is based solely on your written outline.

## 8. Next steps

M1's documentation half is done (`BASELINE_METRICS.md`). M2's merge pipeline is built,
unit-tested, and smoke-tested end-to-end against one real source (`DATA_DISTRIBUTION.md`).
**Everything else is now blocked on the same single thing: a Roboflow API key**
(Open Question #3). Once that's added as a secret, the very next action is:

```
python scripts/build_unified_dataset.py \
  --source construction_site_safety=<path-to-exported-base-dataset> \
  --source ppe_combined_model=<path-to-exported-combined-model> \
  --source hard_hat_universe=<path-to-exported-hard-hat-universe> \
  --source ultralytics_construction_ppe=<path-to-construction-ppe> \
  --out data/unified
```

— no further code changes needed for the merge itself. What *would* still need doing
after that: (a) flip `status="pending_export"` to `"confirmed"` in
`src/data/label_schema.py` for the two Roboflow sources once their real `data.yaml`
class lists are verified against what's currently there (assembled from public research,
not a real export), (b) IoU-based negative synthesis for any classes the Combined Model
export turns out to lack explicit negatives for, and (c) re-generate
`DATA_DISTRIBUTION.md` with the full multi-source table.
