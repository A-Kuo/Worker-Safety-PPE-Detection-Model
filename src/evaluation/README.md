# src/evaluation

Implements the math from `docs/PROJECT_PLAN.md` Section 4 (M3) as real, unit-tested
code — not just formulas in a doc. Everything here is pure Python (only depends on
`pillow`/`scikit-learn`, no `torch`/`ultralytics` required), so it runs in any
environment, including this one (no GPU).

- `boxes.py` — IoU + greedy detection-to-ground-truth matching (the foundation
  every metric below depends on).
- `metrics.py` — per-class precision/recall/F1 and a confusion-matrix
  aggregator across a whole dataset.
- `calibration.py` — Expected Calibration Error (ECE) and Brier score.
- `threshold_optimization.py` — best-F1 threshold vs. a minimum-recall-
  constrained threshold, for safety-critical classes where a false negative
  is worse than a false positive.
- `yolo_adapter.py` — glue to run this against a real trained Ultralytics
  model: label-file parsing and coordinate conversion are unit-tested here;
  the one function that needs `ultralytics`/`torch` installed
  (`run_yolo_predictions`) is lazily imported so the rest of the module still
  works without them. Run the actual prediction step from
  `notebooks/train_on_kaggle_or_colab.ipynb` (GPU required).

38 unit tests across `tests/test_boxes.py`, `tests/test_metrics.py`,
`tests/test_calibration.py`, `tests/test_threshold_optimization.py`, and
`tests/test_yolo_adapter.py` cover every module here using synthetic data —
this code has not yet been run against real model predictions (no trained
model exists yet), but the math itself is exercised and correct.
