# End-to-end pipeline commands

Run from the **repository root**. Downloads need `ROBOFLOW_API_KEY`. **Train E0–E4 on Colab or Kaggle** ([docs/compute.md](../docs/compute.md)). Local 8GB is for remap, baseline val, demo, and `--batch 8` smokes. Do not commit raw images or experiment weights. Boots are out of this cycle.

```bash
# 0) Env
python -m pip install -r requirements.txt
python -m pip install -e .   # enables `from ppe...`

# 1) Download (omit --execute for dry-run)
python scripts/download_datasets.py --dry-run
python scripts/download_datasets.py --execute

# 2) Remap onto unified schema (never merge Construction into Combined)
python scripts/remap_labels.py --source data/raw/combined --out data/processed/combined --mapping combined
python scripts/remap_labels.py --source data/raw/hardhat --out data/processed/hardhat --mapping hhu
python scripts/remap_labels.py --source data/raw/construction --out data/processed/construction --mapping construction

# 3) Stratified 12k subset for E0–E3
python scripts/make_subset.py --source data/raw/combined --out data/raw/combined_12k --n 12000 --seed 42

# 4) Distribution analysis (before training)
python scripts/analyze_distribution.py

# 5) Train on Colab/Kaggle (see notebooks/train_colab_kaggle.ipynb).
# Local 8GB smoke only:
python scripts/train.py --exp e0_n --dry-run
python scripts/train.py --exp e0_n --batch 8 --device 0

# 6) Eval + calibration (point --weights at your run)
python scripts/eval.py --weights runs/train/e0_n/weights/best.pt
python scripts/eval_cross_domain.py --weights runs/train/e4_full44k/weights/best.pt
python scripts/calibrate.py --weights runs/train/e4_full44k/weights/best.pt

# 7) Export + latency
python scripts/export_onnx.py --weights runs/train/e4_full44k/weights/best.pt
python scripts/benchmark.py --weights runs/train/e4_full44k/weights/best.pt

# 8) Demo (second terminal for UI)
python -m pip install -r app/requirements.txt
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
streamlit run app/ui/streamlit_app.py
```

Inherited Construction baseline audit (no Combined download required if weights exist):

```bash
python scripts/eval_baseline.py
```

Docs: [docs/baseline.md](../docs/baseline.md), [docs/experiments.md](../docs/experiments.md), [docs/data_distribution.md](../docs/data_distribution.md), [app/README.md](../app/README.md).
