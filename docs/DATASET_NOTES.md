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

**Confirmed as an M2 source (2026-08-16).** Blocked on a Roboflow API key — the
`roboflow` Python package refuses all access without one (`ValueError: A valid API key
must be provided`), confirmed by actually attempting the call this session; this is not
a paid-tier limitation, it's a hard requirement for every export, even fully public ones.

- URL: `https://universe.roboflow.com/roboflow-universe-projects/personal-protective-equipment-combined-model`
- Size: **44,002 images**. Version `/4` ("resize640_allClasses_noAugs", generated
  Dec 3, 2022) confirmed split: **70% train (30,765) / 20% valid (8,814) / 10% test
  (4,423)**. (Version `/8` also exists per earlier research; re-check which version to
  export from once we have API access — `/4`'s "allClasses" framing in its own name
  suggests it may be the more complete class list.)
- Classes observed in the public API/UI snippet (list is not exhaustively confirmed —
  the page truncates with "+ custom", meaning more classes likely exist than shown):
  `Mask, Goggles, Person, Hardhat, Ladder, Safety Vest, Gloves` (at minimum). A
  third-party repo's README (`NamHoKi/PPE-Detection-for-Construction-Site-Safety`) links
  to a browse query for `class:"NO-Safety Vest"` against this exact project, confirming
  it **also has at least one explicit negative class** (`NO-Safety Vest`) — so it isn't
  purely positive-only the way earlier research suggested.
  **Action:** pull the actual `data.yaml` from a real export to get the authoritative
  full class list and per-class counts before finalizing the mapping table below;
  `src/data/label_schema.py`'s `SOURCES["ppe_combined_model"]` is marked
  `status="pending_export"` for exactly this reason.
- This dataset is the best candidate to satisfy the "Safety Goggles" and "gloves"
  requirements in the outline's unified schema, since it already contains `Goggles` and
  `Gloves` classes that the base dataset lacks.

## 4. Hard Hat Universe (link #2 in the outline)

**Confirmed as an M2 source (2026-08-16).** Same API-key blocker as above.

- URL: `https://universe.roboflow.com/universe-datasets/hard-hat-universe-0dy7t`
- Size: confirmed exact for version **26** ("no_nulls_plain", generated Jun 28, 2022):
  **7,034 total images**, split **70% train (4,912) / 20% valid (1,414) / 10% test
  (708)**.
- Classes (6, now fully confirmed via the Roboflow Universe UI text): `head, helmet,
  person, hi-viz, hi-viz helmet, hi-viz vest`. (Earlier research only surfaced 5 of the
  6; the missing one was the generic `hi-viz` class, without a body-part qualifier.)
- Framing (from the dataset's own description): annotations explicitly include bare
  `head` and `person`-without-helmet cases "for when an individual may be present without
  a hard hat" — i.e., **`head` is effectively the implicit negative for `helmet`** in
  this dataset, which is exactly the kind of "implicit vs. explicit negative" mismatch
  called out in the Project Plan's risk register. Current mapping decision (in
  `src/data/label_schema.py`): `head` → `no_helmet` (treated as this dataset's explicit
  negative, not synthesized), generic `hi-viz` → `vest` (closest unified class for an
  ambiguous "wearing hi-vis clothing generally" tag — flagged for re-review with real
  images, the same way the Construction-PPE `none` ambiguity below was resolved).
- A related paper (Vision-Based Construction Safety Monitoring, MDPI 2024) used a
  *different* derived dataset with 5 classes (`person, hard hat, boots, vest, robodog`)
  and cites `hard-hat-universe-0dy7t` only as an external classification test set (214
  images, hand-labeled safe/unsafe) — useful prior art for the M4 cross-domain
  evaluation methodology, not a dataset to import directly.

## 5. Ultralytics Construction-PPE — added this session as an interim/supplementary source

**Not one of your two linked datasets** — added because it's downloadable with no API
key (`https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip`,
178.4 MB) and already uses almost exactly your target unified taxonomy, letting the merge
pipeline be built and tested against real data today instead of staying purely
theoretical until Roboflow access is sorted out. This is a genuine addition, not a
silent substitution for either of your two links — both remain the primary named M2
sources and are still queued up in `src/data/label_schema.py`, just blocked.

- Source: official Ultralytics dataset asset, documented at
  `docs.ultralytics.com/datasets/detect/construction-ppe` and configured via
  `ultralytics/cfg/datasets/construction-ppe.yaml` in the `ultralytics/ultralytics` repo.
  **License: AGPL-3.0** (see risk register in `PROJECT_PLAN.md` — a non-issue for
  local/portfolio use, relevant only if a live public API is ever stood up).
- Size: **1,416 images** (1,132 train / 143 val / 141 test).
- 11 classes: `helmet, gloves, vest, boots, goggles, none, Person, no_helmet, no_goggle,
  no_gloves, no_boots`.
- **Resolved ambiguity**: the class literally named `"none"` is not a generic/background
  tag to drop. Rendering this dataset's own bounding boxes onto its own images (e.g.
  `image1117.jpg`: a `no_helmet` box over the head region and a `none` box over the torso
  region, in the same photo) shows `none` is consistently placed over the torso of a
  person not wearing a hi-vis vest — i.e. it's this dataset's (oddly-named) `no_vest`
  negative. Mapped accordingly in `src/data/label_schema.py`; see
  `tests/test_label_schema.py::test_construction_ppe_none_class_maps_to_no_vest`.
- Real class-instance distribution and a real dedup run against this dataset are in
  `docs/DATA_DISTRIBUTION.md` — including a genuine, unplanned finding that ~47% of this
  dataset's own detected duplicate clusters span more than one of train/val/test
  (likely consecutive video frames), suggesting some pre-existing leakage independent of
  anything we're merging in.

## 6. Datasets named in the outline but not directly linked — resolved

The outline's Step 2 named **"Safety Goggles"** and **"PPE‑Additions‑2314"** as datasets
to import. **Resolved (2026-08-16, your confirmation):** use the two Universe links
provided (Combined Model + Hard Hat Universe) as the actual M2 sources; no separate
datasets by those exact names needed to be tracked down.

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
