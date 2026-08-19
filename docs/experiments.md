# Experiments

Fixed data, split, and seed. One factor changes per run on the stratified Combined 12k subset, then a single confirmation run on the full 44k set.

Combined 14-class mAP is not comparable to the inherited Construction 10-class baseline (mAP50 = 0.809). Comparisons run on `SHARED_EVAL_CLASSES` only: helmet, vest, mask, person, cone, and their `no_*` partners.

Safety-critical classes for false-negative analysis: `no_helmet`, `no_vest`, `no_goggles`, `no_mask`.

## Protocol

| Stage | Data | Model | Epochs | Purpose |
|---|---|---|---|---|
| E0-E3 grid | Combined 12k subset (`data/raw/combined_12k`) | see table | 100 or early stop | isolate size, loss, augmentation |
| E4 confirm | Combined full 44k | YOLOv8n | 50 + early stop | shipped detector |
| E5 (optional) | only if E1 and E4 finished | YOLOv8m | n/a | not scheduled |

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
| E2 | `fl_gamma=1.5` | Imbalance / FN on `no_*` | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run | Ultralytics focal |
| E3 | Stronger augs (blur, brightness, occlusion-ish crop) | Industrial cameras | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run | same subset/seed |
| E4 | YOLOv8n 50e on full 44k | Confirm subset did not lie | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run | shipped weights |

## Shared-class comparison vs Construction baseline

Construction baseline (Snehil Sanyal, 10-class, val n=114): global mAP50 = 0.809, mAP50-95 = 0.507, P = 0.900, R = 0.731. Per-class numbers need `scripts/eval_baseline.py` with images on disk.

| Model | helmet | no_helmet | vest | no_vest | mask | no_mask | person | cone | shared mAP50 |
|---|---|---|---|---|---|---|---|---|---|
| Construction YOLOv8n (inherited) | pending val pass | pending val pass | pending val pass | pending val pass | pending val pass | pending val pass | pending val pass | pending val pass | n/a (10-class global 0.809) |
| E0 Combined 12k | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run |
| E4 Combined 44k | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run |

## Cross-domain (same E4 weights)

Reading HHU `head` as `no_helmet` is an assumption, so it is scored separately and kept out of Combined metrics.

| Domain | Split | mAP50 | mAP50-95 | P | R | classes |
|---|---|---|---|---|---|---|
| Combined test | test | pending training run | pending training run | pending training run | pending training run | 14 unified |
| Construction (mapped shared) | test | pending training run | pending training run | pending training run | pending training run | `SHARED_EVAL_CLASSES` |
| HHU remapped | test | pending training run | pending training run | pending training run | pending training run | helmet / vest / person / no_helmet |

Once the numbers exist, the open question is one general Combined model against a helmet specialist. The specialist is out of scope unless E4 finishes early.

## Calibration and `no_*` thresholds

Target recall on violation classes is 0.90, reported alongside the precision it costs. `scripts/calibrate.py` writes `results/analysis/calibration.json`.

| Class | ECE | Brier | thr at R >= 0.90 | precision at that thr | precision cost (1-P) | best R if 0.90 missed |
|---|---|---|---|---|---|---|
| all detections | pending training run | pending training run | n/a | n/a | n/a | n/a |
| no_helmet | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run |
| no_vest | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run |
| no_goggles | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run |
| no_mask | pending training run | pending training run | pending training run | pending training run | pending training run | pending training run |

## Latency (640x640 and 960x540)

Fill from `scripts/benchmark.py` after `scripts/export_onnx.py`.

| Backend | size | latency ms | FPS | RSS MB | VRAM MB |
|---|---|---|---|---|---|
| PyTorch | 640x640 | pending training run | pending training run | pending training run | pending training run |
| ORT | 640x640 | pending training run | pending training run | pending training run | pending training run |
| PyTorch | 960x540 | pending training run | pending training run | pending training run | pending training run |
| ORT | 960x540 | pending training run | pending training run | pending training run | pending training run |

## External reference (not this work)

[Hexmon/vyra-yolo-ppe-detection](https://huggingface.co/Hexmon/vyra-yolo-ppe-detection) is a YOLOv8m trained on Combined v4. Cite it as an external reference, not as a result from this repo.
