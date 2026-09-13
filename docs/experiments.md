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
| E0 | YOLOv8n default | Unified baseline | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run | 12k subset |
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
| E0 Combined 12k | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run |
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
