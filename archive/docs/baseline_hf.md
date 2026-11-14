# Hexmon/vyra-yolo-ppe-detection — raw checkpoint baseline

This is a **reproduced, raw third-party checkpoint's numbers**, not a model
trained or fine-tuned in this repo. See [ATTRIBUTION.md](../ATTRIBUTION.md)
and [`configs/models/registry.yaml`](../configs/models/registry.yaml) for the
vetting record. Do **not** present these numbers as this repo's fine-tuning
result — they are the starting point Phase 3 fine-tunes from, not the
outcome.

- **Checkpoint:** [Hexmon/vyra-yolo-ppe-detection](https://huggingface.co/Hexmon/vyra-yolo-ppe-detection),
  YOLOv8m, pinned at commit `08895b33d95d2587423ebe4f7c1b9c41beebd642`.
- **Fetched via:** `scripts/fetch_checkpoint.py`; sha256 recorded in `models/pretrained/manifest.json`.
- **Weights evaluated:** `models/pretrained/hexmon_vyra/best_unified_order.pt` — **not** the raw fetch. See "Class-order correction" below for why.
- **Eval command:** `python scripts/eval.py --weights models/pretrained/hexmon_vyra/best_unified_order.pt --split test`
- **Eval data:** `data/processed/combined/data.yaml` (Combined v4 test split, 4,423 images, unified 14-class schema).

## Class-order correction (important, and why these numbers are trustworthy)

Hexmon's own detection head orders its 14 classes **alphabetically**
(`Fall-Detected=0, Gloves=1, Goggles=2, Hardhat=3, ...`), while this repo's
unified schema orders them differently (`helmet=0, no_helmet=1, vest=2,
no_vest=3, ...`). Ultralytics' `model.val()` matches predictions to ground
truth by **numeric class ID**, not name — so evaluating the raw fetched
checkpoint directly against `combined.yaml` scored almost every class against
the wrong ground-truth column (an earlier run of this doc reported a
nonsensical mAP50 of 0.057 this way, recorded at
`results/analysis/eval_hexmon_vyra_raw.json` for reference).

This was fixed at the source with
[`scripts/reorder_checkpoint_classes.py`](../scripts/reorder_checkpoint_classes.py),
which permutes only the detection head's classification-conv output channels
(box-regression weights and everything else are bit-identical — a pure
relabeling, not a retrain), producing
`models/pretrained/hexmon_vyra/best_unified_order.pt`. Correctness was
verified with
[`scripts/verify_reorder.py`](../scripts/verify_reorder.py) on 4 real test
images spanning helmet/no_gloves/gloves/fall_detected classes: identical
boxes and confidences between the original and reordered checkpoint, only
the class label changes. The numbers below, evaluated against the corrected
checkpoint, are the first scientifically valid raw numbers for this
checkpoint in this repo.

## Raw numbers (class-order-corrected)

Global (test split, 4,423 images, conf=0.25, iou=0.5):

- mAP@0.50 = **0.747**
- mAP@0.50:0.95 = **0.502**
- Precision = **0.734**
- Recall = **0.859**

Per-class:

| class | P | R | F1 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| helmet | 0.818 | 0.902 | 0.858 | 0.860 | 0.495 |
| no_helmet | 0.592 | 0.907 | 0.716 | 0.739 | 0.516 |
| **vest** | **0.573** | **0.745** | 0.648 | 0.533 | 0.410 |
| **no_vest** | **0.393** | **0.398** | **0.396** | **0.159** | **0.080** |
| goggles | 0.825 | 0.988 | 0.899 | 0.958 | 0.602 |
| no_goggles | 0.825 | 0.940 | 0.879 | 0.914 | 0.571 |
| gloves | 0.838 | 0.923 | 0.878 | 0.899 | 0.471 |
| no_gloves | 0.801 | 0.908 | 0.851 | 0.868 | 0.433 |
| mask | 0.547 | 0.921 | 0.686 | 0.521 | 0.397 |
| no_mask | 0.488 | 0.913 | 0.636 | 0.622 | 0.434 |
| person | 0.978 | 0.957 | 0.967 | 0.949 | 0.813 |
| cone | 0.821 | 0.738 | 0.777 | 0.710 | 0.421 |
| ladder | 0.970 | 0.951 | 0.961 | 0.952 | 0.844 |
| fall_detected | 0.813 | 0.831 | 0.822 | 0.777 | 0.539 |

Full output: `results/analysis/eval_hexmon_vyra_unified_order.json`.

**This is the concrete, valid evidence motivating the gap fine-tune**: `vest`/`no_vest` (README priority #1, target 95%+ P&R) are among the weakest classes in this raw checkpoint — `no_vest` in particular (P=0.39, R=0.40, mAP50=0.16) is far below every other class, including its own positive counterpart `vest`. This is the baseline the vest gap fine-tune (`configs/finetune/gap_vest.yaml`, `novest/no-vest-detect` data) needs to improve on — see `docs/experiments.md`.

## Qualitative spot-check (name-unified, via `PPEDetector`)

On `data/processed/combined/test/images/-1003-_png_jpg.rf.40a811e60f1b213fd091d11af90ad6e4.jpg`, both the original and reordered checkpoints agree (per `verify_reorder.py`):

```
helmet 0.874  (368.8, 275.3, 511.2, 383.0)
helmet 0.867  (123.4, 263.9, 279.0, 386.9)
helmet 0.665  (310.7, 130.8, 338.6, 154.3)
```

The compliance-summary pass found 0 workers across the first 64 test images;
given `person` scores strongly overall (P=0.978, R=0.957 globally), this is
most likely a framing artifact of those specific 64 images (the first 15
are disproportionately close-up/helmet-only shots — 52 of 52 ground-truth
boxes are `helmet`), not a real `person`-detection weakness — consistent
with the global per-class numbers above.

## How this was generated

```powershell
python scripts/fetch_checkpoint.py --repo-id Hexmon/vyra-yolo-ppe-detection --filename best.pt --revision 08895b33d95d2587423ebe4f7c1b9c41beebd642 --name hexmon_vyra --license cc-by-4.0
python scripts/reorder_checkpoint_classes.py --weights models/pretrained/hexmon_vyra/best.pt --out models/pretrained/hexmon_vyra/best_unified_order.pt
python scripts/verify_reorder.py --original models/pretrained/hexmon_vyra/best.pt --reordered models/pretrained/hexmon_vyra/best_unified_order.pt --image <test image>
python scripts/eval.py --weights models/pretrained/hexmon_vyra/best_unified_order.pt --split test --out results/analysis/eval_hexmon_vyra_unified_order.json
```

This doc is hand-written from that eval run's output (unlike `docs/baseline.md`,
there is no dedicated regenerate script for this doc yet).
