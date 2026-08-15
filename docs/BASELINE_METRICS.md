# Baseline Metrics — Milestone M1

Status: **Documentation half of M1 complete** (documented from the upstream repo's
existing `results/` folder, per your Step 1 instruction). **Independent retraining
("reproduce") is not yet done** — see §4, this cloud agent VM has no GPU and no
`torch`/`ultralytics` installed, so a from-scratch 100-epoch run isn't feasible here.

Source confirmed by you: **[`snehilsanyal/Construction-Site-Safety-PPE-Detection`](https://github.com/snehilsanyal/Construction-Site-Safety-PPE-Detection)**
— the "existing local files" project. Cloned (shallow) into this session purely for
inspection; nothing was retrained. All numbers below are read directly from that repo's
committed `results/results.csv` and `results/*.png` plots (copied into
`docs/assets/baseline/` for reference — original license: dataset CC BY 4.0, repo has no
separate code license file).

## 1. Dataset (as actually configured in the upstream repo)

From `docs/assets/baseline/data.yaml` (copied verbatim from the upstream repo):

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images
nc: 10
names: ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
roboflow:
  workspace: roboflow-universe-projects
  project: construction-site-safety
  version: 28
  license: CC BY 4.0
```

Confirms your cited split (2605/114/82) and 10-class label set exactly. **Important:**
the actual image files are **not** committed to the GitHub repo — only the yaml configs
and pre-computed outputs are. To retrain or even just re-run validation, images must be
pulled from the Kaggle mirror (`snehilsanyal/construction-site-safety-image-dataset-roboflow`)
or re-exported from Roboflow (`roboflow-universe-projects/construction-site-safety`,
version 28).

## 2. Global metrics (from `results.csv`, final epoch of the logged 100-epoch run)

| Metric | Value |
|---|---|
| Epoch | 99 (0-indexed, i.e. the 100th epoch) |
| Precision | 0.8999 |
| Recall | 0.7311 |
| **mAP@0.5** | **0.8088** (0.809 at epoch 99; the PR-curve plot below independently reports **0.810** for the same run — consistent within rounding) |
| **mAP@0.5:0.95** | **0.5071** |
| Peak F1 (all classes) | **0.81 at confidence threshold 0.488** (from `F1_curve.png`) |

Cross-check: `F1 = 2PR/(P+R)` using the epoch-99 P/R above gives `2×0.8999×0.7311 /
(0.8999+0.7311) ≈ 0.807`, matching the plotted peak F1 of 0.81 — the two independently
generated artifacts agree, which is a reasonable sanity check that these are real,
internally-consistent training outputs rather than mismatched files.

Training curves (loss + metrics vs. epoch) are in `docs/assets/baseline/results.png` —
precision/recall/mAP curves are still trending upward at epoch 99/100 without a clear
plateau, meaning **more epochs likely would have improved this baseline further**; the
upstream author capped training at 100 epochs. Worth revisiting when we retrain.

## 3. Per-class metrics

### 3a. Per-class AP@0.5 (from `docs/assets/baseline/PR_curve.png`)

| Class | AP@0.5 | Instances (train, from `labels.jpg`) |
|---|---|---|
| Mask | 0.918 | ~1,650 |
| machinery | 0.936 | ~5,250 |
| Safety Vest | 0.907 | ~3,050 |
| Safety Cone | 0.872 | ~3,400 |
| Hardhat | 0.856 | ~3,150 |
| Person | 0.832 | ~9,550 |
| NO-Safety Vest | 0.778 | ~3,950 |
| NO-Hardhat | 0.731 | ~2,300 |
| NO-Mask | 0.669 | ~3,100 |
| vehicle | 0.601 | ~1,550 |
| **All classes (mAP@0.5)** | **0.810** | — |

Note the instance counts are read off the `labels.jpg` bar chart (approximate, to the
nearest ~50), not an exact table — exact counts should come from parsing the label
files directly once we have the actual dataset (see §4). Even at this resolution, the
imbalance is clear: `Person` (~9,550 instances) dominates by ~6× over `Mask` and
`vehicle` (~1,550–1,650 each), and this maps directly onto the per-class AP ranking —
`vehicle` and `NO-Mask` are the two worst-performing classes, and they're also two of
the three least-frequent classes. This is exactly the kind of class-imbalance signal
Step 3's focal-loss experiment is meant to address.

### 3b. Per-class recall proxy (from `docs/assets/baseline/confusion_matrix.png`, diagonal)

Ultralytics' normalized confusion matrix (columns = ground truth, rows = predicted;
diagonal ≈ recall at the default operating point, background row = false-negative rate):

| Class | Diagonal (≈ recall) | Confused most with |
|---|---|---|
| machinery | 0.93 | background (0.09) |
| Safety Cone | 0.91 | background (0.06) |
| Mask | 0.90 | background (0.02) |
| Person | 0.80 | background (0.29) |
| Safety Vest | 0.78 | NO-Safety Vest (0.01), background (0.03) |
| Hardhat | 0.76 | background (0.06) |
| NO-Safety Vest | 0.70 | Safety Vest (0.07), background (0.12) |
| NO-Mask | 0.66 | background (0.34) |
| NO-Hardhat | 0.62 | background (0.36) |
| vehicle | 0.57 | background (0.15) |

The `NO-*` negative classes and `vehicle` have the highest background-confusion rates
(0.34–0.36 for `NO-Mask`/`NO-Hardhat`) — i.e., the model's biggest failure mode on this
baseline is **missing the negative/absence classes entirely** (false negatives against
background), not confusing them with the wrong PPE class. This is precisely the failure
mode your Step 3 "reduce false negatives for safety-critical classes" calibration/
threshold-tuning work should target directly.

### 3c. Per-class precision/recall/F1 at a single fixed threshold

Not available from the committed plots at the individual-class level with numeric
precision (the P/R/F1-vs-confidence curves show 10 overlapping lines that are only
reliably readable for the aggregate "all classes" line, not per-class values at a
specific point). **This is a real gap** — closing it exactly requires running
`yolo val model=best.pt data=data.yaml` ourselves against the actual validation images.
That's a CPU-feasible task (inference on 114 images, not training), unlike full
retraining — see §4 as the concrete next step.

## 4. What "reproduce" actually requires (blocked in this environment)

This cloud agent VM has no GPU and no ML libraries installed (`torch`/`ultralytics` not
present). Two distinct sub-tasks remain, with different feasibility:

1. **Exact numeric per-class P/R/F1** (closing the gap in §3c): requires only
   *inference*, not training — running the already-trained `best.pt` (6MB, available in
   the upstream repo) against the 114 validation images. This is realistically
   CPU-feasible in a follow-up session once we can pull the actual images (Kaggle API
   key or Roboflow API key needed — see `PROJECT_PLAN.md` open questions).
2. **True from-scratch reproduction** (retraining YOLOv8n for 100 epochs to verify the
   author's numbers are reproducible, not just copy them): needs a GPU. The original run
   used a Kaggle P100. Recommend running this on Kaggle or Colab (free-tier GPU) using a
   training script we prepare here, then pasting the resulting `results/` folder back
   into this repo — this cloud agent environment is not the right place for GPU training.

## 5. Attribution

Dataset: *Construction Site Safety Image Dataset*, Roboflow Universe
(`roboflow-universe-projects/construction-site-safety`, version 28), license CC BY 4.0.
Baseline training run and result artifacts: `snehilsanyal/Construction-Site-Safety-PPE-Detection`
(GitHub). Artifacts copied into `docs/assets/baseline/` for reference/documentation
purposes only.
