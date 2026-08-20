# End-to-end pipeline commands

Run from the repository root. Downloads need `ROBOFLOW_API_KEY`, and training and eval want a GPU to finish in reasonable time. Raw images and experiment weights stay out of git.

```bash
# 0) Env
python -m pip install -e ".[torch,app]"   # drop `torch` on a device that only runs ONNX

# 1) Download (omit --execute for dry-run)
python scripts/download_datasets.py --dry-run
python scripts/download_datasets.py --execute

# 2) Remap onto the unified schema (Construction is never merged into Combined)
python scripts/remap_labels.py --source data/raw/combined --out data/processed/combined --mapping combined
python scripts/remap_labels.py --source data/raw/hardhat --out data/processed/hardhat --mapping hhu
python scripts/remap_labels.py --source data/raw/construction --out data/processed/construction --mapping construction

# 3) Stratified 12k subset for E0-E3
python scripts/make_subset.py --source data/raw/combined --out data/raw/combined_12k --n 12000 --seed 42

# 4) Distribution analysis (before training)
python scripts/analyze_distribution.py

# 5) Train grid entry (E0); swap --exp for e1_s / e2_focal / e3_augs / e4_full44k
python scripts/train.py --exp e0_n --dry-run
python scripts/train.py --exp e0_n

# 6) Eval + calibration (point --weights at your run)
python scripts/eval.py --weights runs/train/e0_n/weights/best.pt
python scripts/eval_cross_domain.py --weights runs/train/e4_full44k/weights/best.pt
python scripts/calibrate.py --weights runs/train/e4_full44k/weights/best.pt

# 7) Export for the NPU, quantize, then measure
python scripts/export_onnx.py --weights runs/train/e4_full44k/weights/best.pt --imgsz 640 \
    --out models/best.onnx
python scripts/quantize_onnx.py --model models/best.onnx --calibration data/calib
ppe devices
ppe bench --weights models/best.int8.onnx --json

# 8) Service and UI (second terminal for the UI)
ppe serve --port 8000
streamlit run app/ui/streamlit_app.py
```

Auditing the inherited Construction baseline needs no Combined download, only the weights:

```bash
python scripts/eval_baseline.py
```

Docs: [NPU runtime](../docs/npu_runtime.md), [baseline](../docs/baseline.md), [experiments](../docs/experiments.md), [data distribution](../docs/data_distribution.md), [edge runtime](../docs/edge_runtime.md), [service and UI](../app/README.md).
