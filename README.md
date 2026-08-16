# Unified Multi-Domain PPE Compliance Detection

Using Roboflow, fine-tuned model presentation on CV — an end-to-end industrial PPE
detection and compliance-analysis project built on YOLOv8 and Roboflow Universe
datasets. This is a portfolio/demonstration project, not a production or
commercial system.

**Status: M1 (baseline documentation) done; M2 (dataset merge) pipeline built,
unit-tested, and smoke-tested against one real source; blocked on a Roboflow API key
for the two primary named sources.** No training has run yet (this dev environment has
no GPU). Start here:

- [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) — the full engineering plan: milestone
  breakdown (M0–M6) with real status per milestone, feasibility/scope check, unified
  label taxonomy, metric/math definitions (mAP, focal loss, ECE, Brier score, threshold
  optimization), repository structure, and a risk register.
- [`docs/DATASET_NOTES.md`](docs/DATASET_NOTES.md) — grounded, source-verified facts
  about each dataset in scope (Construction Site Safety, Personal Protective Equipment —
  Combined Model, Hard Hat Universe, plus Ultralytics Construction-PPE added as an
  interim source) and the source-class → unified-class label mapping.
- [`docs/BASELINE_METRICS.md`](docs/BASELINE_METRICS.md) — real baseline metrics pulled
  from the upstream repo's own training run (mAP@0.5 = 0.810, per-class breakdown,
  confusion matrix).
- [`docs/DATA_DISTRIBUTION.md`](docs/DATA_DISTRIBUTION.md) — real output from running
  `scripts/build_unified_dataset.py` end to end, including a genuine finding: dedup
  caught train/val/test leakage within a single official dataset.

## Planned scope (see PROJECT_PLAN.md for details)

1. Reproduce a YOLOv8n baseline on the original Construction Site Safety dataset.
2. Merge in additional Roboflow Universe PPE datasets under a unified label schema
   (helmet/vest/goggles/gloves/boots/mask + `no_*` negatives + scene classes).
3. Run backbone/loss/augmentation experiments with full metric, confusion-matrix, and
   calibration (ECE/Brier score) reporting, including threshold tuning for
   safety-critical classes.
4. Evaluate cross-domain robustness across the merged data sources.
5. Ship a FastAPI inference service (batch + streaming) with a demo UI, plus ONNX/INT8
   export and edge-latency benchmarks.
6. *(Stretch)* Add a CLIP/OpenCLIP layer for PPE-compliance captioning and text-based
   frame retrieval.

Every milestone's real, measured results will be documented under `docs/` as they're
produced — numbers in this README and in `docs/PROJECT_PLAN.md` before that point are
plans/targets, not results.
