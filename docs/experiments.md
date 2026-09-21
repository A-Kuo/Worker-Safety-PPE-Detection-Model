# Experiments

Fixed data, split, and seed. Change **one** factor at a time on the stratified Combined 12k subset, then confirm once on the full 44k set.

Do **not** compare Combined 14-class mAP to the inherited Construction 10-class baseline (mAP50 = 0.809). Compare only on `SHARED_EVAL_CLASSES` (helmet / vest / mask / person / cone and their `no_*` partners).

**Primary product metric:** `vest` / `no_vest` at **95%+** P and R. Helmets next. Goggles ~70% is acceptable. Boots are future work.

Safety-critical classes for false-negative analysis: `no_vest` first, then `no_helmet`, `no_goggles`, `no_mask`.

**Compute:** Colab or Kaggle for the grid; local 8GB GPU only for light jobs ([docs/compute.md](compute.md)).

## Protocol

| Stage | Data | Model | Epochs | Purpose |
|---|---|---|---|---|
| E0–E3 grid | Combined 12k subset (`data/raw/combined_12k`) | see table | 100 or early stop | isolate size / loss / augs |
| E4 confirm | Combined full 44k | YOLOv8n | 50 + early stop | shipped detector |
| E5 (optional) | only if E1 and E4 finished | YOLOv8m | — | not scheduled |

Seed, `imgsz=640`, cosine LR, pretrained Ultralytics weights. Configs live in `configs/train/{e0_n,e1_s,e2_focal,e3_augs,e4_full44k}.yaml`.

```bash
python scripts/train.py --exp e0_n --dry-run
python scripts/train.py --exp e0_n
python scripts/train.py --exp e0_n --resume
python scripts/eval.py --weights runs/train/e0_n/weights/best.pt
python scripts/eval_cross_domain.py --weights runs/train/e4_full44k/weights/best.pt
python scripts/calibrate.py --weights runs/train/e4_full44k/weights/best.pt
```

## Grid (subset)

| ID | Change | Why | mAP50 | mAP50-95 | P | R | no_helmet R | no_vest R | notes |
|---|---|---|---|---|---|---|---|---|---|
| E0 | YOLOv8n default | Unified baseline | 0.666 | 0.406 | 0.695 | 0.757 | 0.598 | 0.204 | 12k subset; 94 epochs in 2.97 h on a T4 (early-stopped, best epoch 74). Test split, `eval.py` defaults (conf 0.25). On the deduplicated test (2,055 imgs) mAP50 is 0.675, so near-duplicates do not inflate the aggregate. `no_vest` is the weakest class (mAP50 0.095; see "Lifting `no_vest`") |
| E1 | YOLOv8s | Accuracy vs FPS | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run | same subset/seed |
| E2 | `cls=1.0` (2x default) | Imbalance / FN on `no_*` | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run | `fl_gamma` doesn't exist in this Ultralytics version (was YOLOv5-era; confirmed empirically) — swapped for the real, wired `cls` loss-gain kwarg |
| E3 | Stronger augs (brightness, occlusion-ish crop) | Industrial cameras | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run | same subset/seed; `blur_prob`/`blur_limit` dropped — never a real Ultralytics arg |
| E4 | YOLOv8n 50e on full 44k | Confirm subset did not lie | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run | shipped weights |

## Shared-class comparison vs Construction baseline

Construction baseline (Snehil Sanyal, 10-class, val n=114): global mAP50 = 0.809, mAP50-95 = 0.507, P = 0.900, R = 0.731. Per-class numbers need `scripts/eval_baseline.py` with images on disk.

| Model | helmet | no_helmet | vest | no_vest | mask | no_mask | person | cone | shared mAP50 |
|---|---|---|---|---|---|---|---|---|---|
| Construction YOLOv8n (inherited) | pending val pass | pending val pass | pending val pass | pending val pass | pending val pass | pending val pass | pending val pass | pending val pass | n/a (10-class global 0.809) |
| Hexmon raw (class-order-corrected) | mAP50 0.860 | mAP50 0.739 | mAP50 0.533 | mAP50 0.159 | mAP50 0.521 | mAP50 0.622 | mAP50 0.949 | mAP50 0.710 | 0.747 (14-class global; see `docs/baseline_hf.md`) |
| gap_vest fine-tuned | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run |
| E0 Combined 12k | mAP50 0.816 | mAP50 0.536 | mAP50 0.471 | mAP50 0.095 | mAP50 0.484 | mAP50 0.466 | mAP50 0.926 | mAP50 0.624 | 0.666 (14-class global, test, `eval.py` defaults) |
| E4 Combined 44k | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run |

## Cross-domain (same E4 weights)

HHU `head` → `no_helmet` is an **assumption** (implicit missing helmet). Score it separately so it does not pollute Combined metrics.

| Domain | Split | mAP50 | mAP50-95 | P | R | classes |
|---|---|---|---|---|---|---|
| Combined test | test | pending training run | pending training run | pending training run | pending training run | 14 unified |
| Construction (mapped shared) | test | pending training run | pending training run | pending training run | pending training run | `SHARED_EVAL_CLASSES` |
| HHU remapped | test | pending training run | pending training run | pending training run | pending training run | helmet / vest / person / no_helmet |

Trade-off to write after numbers exist: one general Combined model vs a helmet-specialist. Do not train the specialist unless E4 finished early.

## Calibration and `no_*` thresholds

Target: recall ≥ 0.90 on violation classes, then report the precision cost. See `scripts/calibrate.py` → `results/analysis/calibration.json`.

| Class | ECE | Brier | thr @ R≥0.90 | precision at that thr | precision cost (1−P) | best R if 0.90 missed |
|---|---|---|---|---|---|---|
| all detections | pending training run | pending training run | — | — | — | — |
| no_helmet | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run |
| no_vest | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run |
| no_goggles | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run |
| no_mask | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run |

## Latency (640×640 and 960×540)

Fill from `scripts/benchmark.py` after `scripts/export_onnx.py`.

| Backend | size | latency ms | FPS | RSS MB | VRAM MB |
|---|---|---|---|---|---|
| PyTorch | 640×640 | pending training run | pending training run | pending training run | pending training run |
| ORT | 640×640 | pending training run | pending training run | pending training run | pending training run |
| PyTorch | 960×540 | pending training run | pending training run | pending training run | pending training run |
| ORT | 960×540 | pending training run | pending training run | pending training run | pending training run |

## Scope note: why E1–E3 weren't run this cycle

The E0–E4 grid below is complete and ready to run (data downloaded/remapped/subset, configs written), but this cycle's priority shifted to a pretrained-checkpoint fine-tuning track instead of a full from-scratch grid search (see `docs/baseline_hf.md` and `configs/finetune/`). **E0 is kept and run** as a genuine from-scratch comparison point; **E1–E3 are deliberately deferred**, not abandoned — the question they'd answer (which architecture/loss/aug variant wins from scratch) is superseded for now by the raw-checkpoint-vs-fine-tuned-vs-E0 comparison, which is the more portfolio-relevant result. Revisit E1–E3 if that comparison shows from-scratch training is worth pursuing further.

## Fine-tuning base checkpoint

[Hexmon/vyra-yolo-ppe-detection](https://huggingface.co/Hexmon/vyra-yolo-ppe-detection) (YOLOv8m, trained on Combined v4) was vetted and adopted as the base checkpoint for gap fine-tuning — see `ATTRIBUTION.md` and `configs/models/registry.yaml` for the vetting record, and `docs/baseline_hf.md` for its raw (pre-fine-tune) numbers on this repo's eval sets. Its raw numbers are an external result, not a model trained in this repo. Use `models/pretrained/hexmon_vyra/best_unified_order.pt` (class-order-corrected via `scripts/reorder_checkpoint_classes.py`, not the raw fetch) for any evaluation or fine-tuning — see `ATTRIBUTION.md` for why.

## Gap fine-tuning: vest/no_vest reliability

Gap fine-tuning targets **vest/no_vest** (README priority #1), not goggles as originally scoped — a concrete dataset was found first ([novest/no-vest-detect v1](https://universe.roboflow.com/novest/no-vest-detect), 998 images, ~1,112 no-vest instances, CC BY 4.0; remapped via `scripts/remap_labels.py --mapping gap_vest` with zero dropped boxes). Config: `configs/finetune/gap_vest.yaml` / `configs/data/gap_finetune.yaml`. Goggles hard-negative fine-tuning stays deferred; `configs/finetune/gap_goggles.yaml` remains as a template. Run: `python scripts/finetune.py --exp gap_vest` (Kaggle/Colab GPU — not run locally yet).

## Lifting `no_vest`

`no_vest` was the weakest class (E0 test mAP50 0.095, recall 0.20; Hexmon 0.159). Diagnosis with `scripts/diagnose_class.py`:

- **The model sees the boxes but scores them low.** At conf >= 0.01 E0 finds ~70% of true `no_vest` boxes; at the 0.25 deployment floor only 24%. Only 0.4% are polarity flips (`vest` predicted instead), 24% are missed entirely.
- **Cause: partial annotation.** Combined merges Roboflow sources that each labeled only some classes. 84% of test images with workers have no `vest`/`no_vest` label at all, and the 108 `no_vest` test images carry no helmet labels. Unlabeled torsos are trained as background, which suppresses `no_vest`.
- **Fine-tuning the 14-class model on vest-annotated images does not fix it**: `no_vest` improves but every class not labeled there is forgotten (helmet 0.82 -> 0.00).

### Threshold trade-off (no code change, `PPE_CLASS_CONF`)

Recall / false `no_vest` boxes per 100 clean images, `eval.py` protocol. False-box rates are pessimistic because clean images contain unlabeled workers.

| `no_vest` conf | E0 | Hexmon |
|---|---|---|
| 0.25 | 0.24 / 3.3 | 0.46 / 7.7 |
| 0.10 | 0.41 / 9.3 | 0.60 / 10.7 |
| 0.05 | 0.52 / 13.7 | 0.68 / 12.3 |

### Fix: a separate 2-class vest specialist

`configs/train/vest_specialist.yaml` trains a 2-class YOLOv8n (`vest`, `no_vest`) only on images where vest status is annotated (Combined train/valid + the external `novest/no-vest-detect` set, minus anything that near-duplicates a Combined test image; `scripts/make_vest_specialist_data.py`). At inference its two classes **replace** the main model's `vest`/`no_vest`; all other classes are untouched (`src/ppe/specialist.py`, opt-in via `PPE_SPECIALIST=<weights>`; `PPE_CLASS_CONF="no_vest=0.10"` sets per-class floors).

### Results

Scored on the 145-image deduplicated vest-annotated test split (43 `no_vest` and 249 `vest` boxes, **so per-class numbers are unstable**), standard mAP protocol (`--conf 0.001 --iou 0.7`, the protocol Ultralytics uses for validation):

| Model | `no_vest` mAP50 | `vest` mAP50 |
|---|---|---|
| E0 (14-class) | 0.373 | 0.856 |
| Hexmon raw | 0.624 | 0.904 |
| 14-class fine-tune PoC | 0.583 | 0.871 |
| Vest specialist (CPU, E0-backbone init, 15 epochs) | 0.796 | 0.913 |

Caveats that matter when reading these:

- **Protocol.** `eval.py` defaults to conf 0.25 / IoU 0.5, a deployment operating point. It truncates the PR curve of under-confident classes: the same models read 0.185 / 0.414 / 0.739 on `no_vest` there. Compare only under one protocol; `scripts/sanity_check.py` uses the standard one.
- **Near-duplicates.** 54% of the 4,423 Combined test images have a near-duplicate (64-bit dHash, <= 6 bits) in train/valid. This does **not** move the aggregate headline (E0 0.666 -> 0.675, Hexmon 0.747 -> 0.753 on the 2,055 clean images), but it does inflate models fine-tuned on the 63%-duplicated vest subset (a `no_vest` fine-tune read 0.752 contaminated vs 0.541 clean). All specialist numbers above use the clean list.
- The CPU checkpoint is a proof of the approach; the shipped specialist comes from the Kaggle GPU run (`EXP = "vest_specialist"` in `notebooks/train_colab_kaggle.ipynb`), to be re-scored the same way.
