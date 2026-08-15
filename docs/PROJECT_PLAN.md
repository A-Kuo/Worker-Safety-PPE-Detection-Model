# Unified Multi-Domain PPE Compliance Detection — Project Plan

Status: **Planning artifact** (no training/serving code written yet).
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

**Action needed from you:** push/commit the local file tree (or point me at the fork URL
you already created) in a follow-up so Step 1's "reproduce baseline exactly" can run
against your actual checked-out `results/` folder rather than a fresh clone of upstream.
Until then, Step 1 below describes how to fork+reproduce from scratch using the verified
upstream repo.

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
your Step 5 — this is what we'll modernize into FastAPI.

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

### **M1 — Fork & reproduce baseline (your Step 1)**

- Objective: exact reproduction of YOLOv8n on the original 2605/114/82 split, with
  numbers you generated yourself and can defend.
- Tasks:
  1. Fork `snehilsanyal/Construction-Site-Safety-PPE-Detection` (or import your existing
     local copy once pushed).
  2. Re-run training with the same hyperparameters (100 epochs, imgsz, batch=16) —
     record exact `ultralytics` version, since YOLOv8 defaults have drifted across
     releases and that alone can shift mAP by a few points.
  3. Pull `results/` (confusion matrix, PR curves, per-class P/R) and `runs/detect/train/`
     outputs into `docs/BASELINE_METRICS.md` verbatim (numbers + plot images), not
     paraphrased.
- Deliverables: `docs/BASELINE_METRICS.md` with: global mAP@0.5, mAP@0.5:0.95; per-class
  P/R/F1 table (all 10 classes); confusion matrix image; PR curve image; environment
  spec (GPU, `ultralytics` version, seed).
- Risks: Kaggle P100 vs. your actual compute (Colab T4/L4, local GPU, CPU-only) will
  shift wall-clock and possibly final metrics slightly — document actual hardware used,
  don't claim P100 numbers if you trained elsewhere.

### **M2 — Expand & normalize datasets (your Step 2)**

- Objective: merge the base dataset with the two Roboflow Universe sources you linked
  into one YOLO-format dataset under the unified schema from Section 3.
- Tasks:
  1. Export `personal-protective-equipment-combined-model` (44,002 images — this is
     large; consider exporting a stratified subset first to keep iteration fast) and
     `hard-hat-universe-0dy7t` (~7,000 images) from Roboflow in YOLOv8 format.
  2. **Cross-dataset deduplication** (perceptual hash / MD5 on images) before merging —
     Universe forks commonly share source photos; skipping this risks train/test leakage
     across the merged splits.
  3. Remap every source label file to the unified taxonomy per the DATASET_NOTES mapping
     table; synthesize negative classes (`no_*`) via IoU-based person↔PPE association
     (a person box with no matched `helmet` box above IoU/containment threshold → emit
     `no_helmet` box at the person's head region — this needs a clear, documented
     heuristic, since it's inventing labels the annotators didn't provide).
  4. Re-split (stratified by dataset source *and* class, not just randomly) into unified
     train/val/test.
  5. Data distribution analysis notebook/script: counts per class × per source dataset ×
     resulting merged split; explicit imbalance report (e.g., `no_goggles` will be rare
     because goggles are rare across all three sources); domain-shift note (construction
     site vs. combined-model's mixed environments vs. hard-hat-universe's workplace
     scenes).
- Deliverables: `data/unified/` (or documented external path) + `configs/data_unified.yaml`;
  `docs/DATA_DISTRIBUTION.md` with counts tables and imbalance/domain-shift callouts;
  a `scripts/build_unified_dataset.py` that is re-runnable end-to-end from raw Roboflow
  exports (so the pipeline is reproducible, not manual).
- Risks (high — this is the hardest step, matching your own assessment): synthesized
  negatives are heuristic and will contain noise; annotation-style mismatch (bbox
  tightness, occlusion handling) between datasets can look like "domain shift" when it's
  actually "labeling-convention shift" — call this out explicitly in the writeup rather
  than conflating the two.

### **M3 — Model experiments & mathematical baseline (your Step 3)**

- Objective: establish YOLOv8n baseline on the *unified* dataset, then run a scoped
  ablation grid with full metric + calibration reporting.
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
│   ├── BASELINE_METRICS.md         # M1 output
│   ├── DATA_DISTRIBUTION.md        # M2 output
│   ├── EXPERIMENTS.md              # M3 output
│   ├── CROSS_DOMAIN_EVAL.md        # M4 output
│   ├── DEPLOYMENT_BENCHMARKS.md    # M5 output
│   └── VLM_LAYER.md                # M6 output
├── scripts/
│   └── build_unified_dataset.py    # M2: reproducible merge pipeline
├── src/
│   ├── data/                       # dataset export/merge/remap utilities
│   ├── training/                   # training entrypoints + experiment configs
│   ├── evaluation/                 # metrics, calibration, threshold tuning
│   ├── serving/                    # FastAPI app + inference wrappers
│   └── vlm/                        # M6 CLIP/OpenCLIP layer
├── notebooks/                      # exploratory EDA, kept out of the critical path
└── tests/                          # unit tests for label remapping, metric functions
```

## 6. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Full experiment grid (Step 3) exceeds available time/compute | High | Delays MVP | Cap grid per Section 0; log skipped cells as future work |
| Synthesized negative labels (`no_goggles`, etc.) are noisy | High | Undermines metric validity | Document heuristic explicitly; manually spot-check a sample; report as a named limitation |
| Cross-dataset image duplication causes train/test leakage | Medium | Inflated/misleading metrics | Perceptual-hash dedup before splitting (M2) |
| Annotation-convention differences mistaken for domain shift | Medium | Wrong conclusions in M4 | Explicitly separate "labeling convention" vs. "visual domain" in writeup |
| `"Safety Goggles"` / `"PPE-Additions-2314"` datasets from your outline don't match the two Universe links you sent | Medium | Step 2 sourced from wrong data | Confirm exact dataset IDs (see Open Questions) before the M2 export step |
| Roboflow export requires an API key/paid export tier for large datasets (44k images) | Medium | Blocks M2 | Confirm `ROBOFLOW_API_KEY` availability; consider a stratified subset export first |
| LoRA-for-CNN-detector claim is technically unusual and could read as buzzword-chasing in a portfolio review | Low–Medium | Credibility | Either implement head-freezing/fine-tuning honestly labeled as "adapter-style," or skip and say so |
| VLM layer scope creep turns a detection project into an unfinished multimodal project | Medium | Dilutes core deliverable | Keep M6 as a strictly separate, clearly-labeled stretch PR after M1–M5 are solid |

## 7. Open questions (need your input before M2 can start for real)

1. Please push or share the local file tree you mentioned (or the fork URL) so M1 can
   reproduce against your actual artifacts instead of a fresh clone.
2. Your outline names "Safety Goggles" and "PPE‑Additions‑2314" datasets specifically,
   but the two URLs you sent are `personal-protective-equipment-combined-model` (which
   does include a `Goggles` class already) and `hard-hat-universe-0dy7t`. Should I treat
   the combined-model as the "goggles" source and hard-hat-universe as the third
   domain, or do you have distinct links for the exact datasets named in your outline?
3. Do you have (or can you provision) a Roboflow API key for programmatic export? The
   combined model has 44,002 images — full export may need a paid plan or a filtered/
   stratified export.
4. What compute do you actually have available (Colab/Kaggle free tier, a local GPU,
   CPU-only)? This determines how much of the M3 experiment grid is realistic this week
   vs. needs trimming further.
5. Can you confirm whether the two attached PDFs (`AustinKuo_FrwrdDepEng_v26...`,
   `AustinKuo_DataEngineerCV_vT_26...`) contain additional constraints (e.g., a specific
   target audience like Matroid) that should reshape prioritization — they weren't
   accessible in this environment, so this plan is based solely on your written outline.

## 8. Next steps

Once you confirm the open questions above (particularly #1 and #2), the next PR should
implement **M0 (scaffold)** for real and **M1 (baseline reproduction)**, since those are
fully unblocked today regardless of the answers. M2 onward is blocked on dataset access
details.
