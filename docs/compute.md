# Compute split

| Job | Where | Why |
|---|---|---|
| Combined 12k grid (E0–E3) | **Colab or Kaggle** (P100 / T4) | ~10–18h per 100-epoch n/s run — session GPUs, not 8GB laptop |
| Combined full 44k (E4) | **Colab or Kaggle** | ~20h+ on P100 at 50 epochs |
| Construction remap, 12k subset, distribution | **Local CPU** | Disk/CPU only |
| Inherited baseline val (~114 images), ONNX export, demo, smoke train | **Local 8GB GPU** | Fits; keep batch 8 if training at all |
| Steel-toe **boots** | **Not this cycle** | Combined has no boot class; future dataset only |

Default train YAMLs use `batch: 16` (Colab/Kaggle). Local 8GB fallback:

```bash
python scripts/train.py --exp e0_n --batch 8 --device 0
```

Prefer Colab **or** Kaggle — not both. Notebook: [`notebooks/train_colab_kaggle.ipynb`](../notebooks/train_colab_kaggle.ipynb).

**Vest-first targets** (product, not a single “accuracy” number):

| Priority | Classes | Bar |
|---|---|---|
| 1 | `vest` / `no_vest` | **95%+** precision and recall |
| 2 | `helmet` / `no_helmet` | next; stretch high 80s–90s |
| 3 | `goggles` / `no_goggles` | **~70% acceptable** (clear lenses vs glasses) |
| Future | boots | ~70% if a boot dataset is added later |
