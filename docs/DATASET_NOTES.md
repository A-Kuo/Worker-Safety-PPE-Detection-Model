# Dataset Notes (grounded research)

These facts were verified via web research on 2026-08-15 against the public GitHub
repos and Roboflow Universe pages referenced in the project outline, since the source
PDFs and local file tree were not accessible in this workspace. Roboflow Universe pages
sit behind Cloudflare bot-checks that block automated fetching, so class lists for the
two Universe datasets are taken from cached search snippets / third-party citations —
**re-verify the exact class list and count in the Roboflow UI (or via the export
metadata `data.yaml`) before building the unified dataset**, since Universe project
versions can change class lists between exports.

**Update (confirmed by you, 2026-08-15):** the "existing local files" project is
**`snehilsanyal/Construction-Site-Safety-PPE-Detection`**, matching the repo already
identified below — no change needed to the base-dataset identification. You also
confirmed `Vinayakmane47/PPE_detection_YOLO` is a secondary reference only ("files not
copied, less inspiration") — Section 2 and `PROJECT_PLAN.md` M5 have been downgraded
accordingly: it's a loose pattern reference for the compliance-messaging API shape, not
something we fork or copy structure from directly.

## 1. Base dataset — Construction Site Safety (Step 1 baseline)

- Source: Roboflow Universe, `roboflow-universe-projects/construction-site-safety`.
- License: **CC BY 4.0**.
- Reference implementation: [`snehilsanyal/Construction-Site-Safety-PPE-Detection`](https://github.com/snehilsanyal/Construction-Site-Safety-PPE-Detection)
  (also mirrored on Kaggle by the same author for easier download).
- Size: **2,801 images**, split `train: 2605 / valid: 114 / test: 82` — matches the
  split named in the outline exactly.
- Classes (10): `Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person,
  Safety Cone, Safety Vest, machinery, vehicle`.
- Note the asymmetry already present upstream: there's a `NO-Hardhat` and `NO-Mask` and
  `NO-Safety Vest`, but **no `NO-Safety Cone`, no goggles/gloves/boots classes at all**
  (positive or negative). This dataset alone cannot fulfill the full unified taxonomy —
  hence Step 2's need to bring in additional sources.
- Baseline training reference (independent reproduction, for plausibility-checking your
  own numbers, not as ground truth): [`Venta02/ppe-detection-yolov8`](https://github.com/Venta02/ppe-detection-yolov8)
  reports YOLOv8n → mAP@0.5 ≈ 0.733, mAP@0.5:0.95 ≈ 0.419, P ≈ 0.904, R ≈ 0.665, on the
  114-image validation set, at a best-F1 operating threshold ≈ 0.51. **The actual
  upstream repo's own committed `results/results.csv` and plots report mAP@0.5 ≈ 0.810,
  mAP@0.5:0.95 ≈ 0.507** at epoch 99/100 — noticeably higher than the independent
  reproduction. Full numbers, per-class breakdown, and the source plots are now in
  `docs/BASELINE_METRICS.md`; treat the discrepancy between the two as evidence that
  even "exact reproduction" is sensitive to `ultralytics` version, exact hyperparameters,
  and possibly which checkpoint/split is being reported — worth calling out explicitly
  when we do our own retraining run.
- **Actual image files are not committed to the GitHub repo** — only `data.yaml`,
  `ppe_data.yaml`, and the two Roboflow-generated README files are present under `data/`.
  To retrain or run inference, the images must come from the Kaggle mirror
  (`snehilsanyal/construction-site-safety-image-dataset-roboflow`) or a fresh Roboflow
  export of `roboflow-universe-projects/construction-site-safety`, version 28.
- **Confirmed (not hypothetical) cross-dataset image overlap risk**: the upstream repo's
  `data/README.dataset.txt` states this dataset's images were themselves *cloned from*
  several other Roboflow Universe projects, including:
  `personal-protective-equipment-combined-model` (the exact dataset behind **link #1 in
  your outline**), plus `people-and-ladders`, `safety-vests`, `excavators-cwlh0`,
  `mit-indoor-scene-recognition` (used only for null/background images), and
  `people-detection-general`. **This means the base dataset and the Combined Model
  dataset almost certainly share source images already** — this elevates the "cross-
  dataset duplication" item in `PROJECT_PLAN.md`'s risk register from a generic caution
  to a confirmed, must-handle risk before merging: run perceptual-hash dedup between the
  base dataset and the Combined Model export specifically, not just as a generic
  best-practice, and check whether any duplicate lands in one dataset's train split and
  the other's test split (the worst-case leakage scenario).

## 2. Deployment reference (Step 5 — secondary inspiration only, confirmed not copied)

- [`Vinayakmane47/PPE_detection_YOLO`](https://github.com/Vinayakmane47/PPE_detection_YOLO) —
  Flask app (`app.py`), same 10-class label set as the base dataset, supports image/video
  upload and a live webcam route. **You confirmed this repo's files were not copied and
  it should get "less inspiration"** — treat it purely as a loose reference for the
  general shape of a YOLOv8 + Flask upload/webcam flow, not as a structural base to fork
  or port code from. M5's FastAPI service should be designed from the endpoint/UX
  requirements in `PROJECT_PLAN.md` directly rather than by porting this repo.
- Other Flask/YOLOv8 PPE apps found during research (even more secondary — useful only as
  prior art for the compliance-messaging pattern, e.g. "Worker 3 — missing helmet and
  vest"): `Sreejith2/PPE_Detection` (React + Flask, per-industry PPE requirement rules,
  `/upload` REST endpoint returning `detected_ppe`/`missing_ppe`/`safety_message`),
  `dinraj910/Construction-Site-Safety-YOLO` (Flask + OpenCV, person↔PPE/machinery
  association via IoU).

## 3. Personal Protective Equipment — Combined Model (link #1 in the outline)

- URL: `https://universe.roboflow.com/roboflow-universe-projects/personal-protective-equipment-combined-model`
- Size: **44,002 images** (version `/8` at time of research) — this is an order of
  magnitude larger than the base dataset; plan for a stratified subset for fast
  iteration before committing to a full export.
- Classes observed in the public API/UI snippet (list is not exhaustively confirmed —
  the page truncates with "+ custom", meaning more classes likely exist than shown):
  `Mask, Goggles, Person, Hardhat, Ladder, Safety Vest, Gloves` (at minimum).
  **Action:** pull the actual `data.yaml` from an export to get the authoritative full
  class list and per-class counts before finalizing the mapping table below.
- This dataset is the best candidate to satisfy the "Safety Goggles" and "gloves"
  requirements in the outline's unified schema, since it already contains `Goggles` and
  `Gloves` classes that the base dataset lacks.

## 4. Hard Hat Universe (link #2 in the outline)

- URL: `https://universe.roboflow.com/universe-datasets/hard-hat-universe-0dy7t`
- Size: **~7,000 images**.
- Classes (6, per the page's "Classes (6)" listing — only 5 were resolvable from the
  cached snippet): `head, helmet, person, hi-viz helmet, hi-viz vest`, plus one
  additional class not captured in the snippet (re-check the export `data.yaml`).
- Framing (from the dataset's own description): annotations explicitly include bare
  `head` and `person`-without-helmet cases "for when an individual may be present without
  a hard hat" — i.e., **`head` is effectively the implicit negative for `helmet`** in
  this dataset, which is exactly the kind of "implicit vs. explicit negative" mismatch
  called out in the Project Plan's risk register.
- A related paper (Vision-Based Construction Safety Monitoring, MDPI 2024) used a
  *different* derived dataset with 5 classes (`person, hard hat, boots, vest, robodog`)
  and cites `hard-hat-universe-0dy7t` only as an external classification test set (214
  images, hand-labeled safe/unsafe) — useful prior art for the M4 cross-domain
  evaluation methodology, not a dataset to import directly.

## 5. Datasets named in the outline but not directly linked

The outline's Step 2 explicitly names **"Safety Goggles"** and **"PPE‑Additions‑2314"**
as datasets to import, but the two URLs actually provided are the Combined Model and
Hard Hat Universe above. These may be:

- The same datasets under different working names (e.g., "Safety Goggles" ≈ the
  `Goggles` class already inside the Combined Model), or
- Distinct Roboflow Universe projects that need their own links — `"2314"` looks like it
  could be a project/version identifier rather than a descriptive name, which isn't
  resolvable via search alone.

**This is flagged as Open Question #2 in `PROJECT_PLAN.md` — confirm before building the
unified dataset**, so M2 doesn't merge the wrong sources.

## Unified label mapping

Working mapping from source classes to the unified taxonomy in `PROJECT_PLAN.md` §3.
Mark `derived` for labels that don't exist explicitly in the source and must be
synthesized (see Project Plan M2, IoU-based person↔PPE association).

| Unified class | Base (Construction Site Safety) | Combined Model | Hard Hat Universe |
|---|---|---|---|
| `helmet` | `Hardhat` | `Hardhat` | `helmet`, `hi-viz helmet` |
| `no_helmet` | `NO-Hardhat` | *derived* | `head` (bare head w/o helmet match) |
| `vest` | `Safety Vest` | `Safety Vest` | `hi-viz vest` |
| `no_vest` | `NO-Safety Vest` | *derived* | *derived* |
| `goggles` | — (not present) | `Goggles` | — (not present) |
| `no_goggles` | — | *derived* | — |
| `gloves` | — (not present) | `Gloves` | — (not present) |
| `no_gloves` | — | *derived* | — |
| `boots` | — (not present in any confirmed source class list) | *check export* | *check export* |
| `no_boots` | — | *derived* | *derived* |
| `mask` | `Mask` | `Mask` | — (not present) |
| `no_mask` | `NO-Mask` | *derived* | — |
| `person` | `Person` | `Person` | `person` |
| `machinery` | `machinery` | *check export* | — |
| `vehicle` | `vehicle` | *check export* | — |
| `cone` | `Safety Cone` | *check export* | — |
| `ladder` | — | `Ladder` | — |

Notes:

- Every "—" cell means the source dataset simply has no annotations for that concept;
  merging does **not** invent detections, it only means that dataset contributes zero
  positive/negative examples for that class, which by itself worsens class imbalance for
  rare classes like `boots` — factor this into the M2 distribution-analysis writeup.
- `boots` has no confirmed source in any of the three datasets researched here. If the
  outline's target schema requires it, we need either a fourth dataset (a dedicated
  "safety boots" Roboflow project) or to drop `boots`/`no_boots` from the MVP taxonomy
  and note it as a known gap.
- Cells marked "*check export*" require pulling the actual `data.yaml`/class list from a
  real Roboflow export rather than the truncated UI snippet — do this at the start of M2
  before writing the remap script, and update this table with the confirmed values.
