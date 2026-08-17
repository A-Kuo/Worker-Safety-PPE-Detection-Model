# Data distribution

Run this analysis **before** any new Combined training. It is the portfolio differentiator:
class imbalance, split sizes, bbox scale, and domain shift (Combined vs HHU vs tiny Construction val).

## How to regenerate

```bash
# 1) Download (requires ROBOFLOW_API_KEY; omit --execute for a dry-run)
python scripts/download_datasets.py --dry-run
python scripts/download_datasets.py --execute

# 2) Remap onto the unified schema (never merge Construction into Combined)
python scripts/remap_labels.py --source data/raw/combined --out data/processed/combined --mapping combined
python scripts/remap_labels.py --source data/raw/hardhat --out data/processed/hardhat --mapping hhu
python scripts/remap_labels.py --source data/raw/construction --out data/processed/construction --mapping construction

# 3) Optional 12k grid subset
python scripts/make_subset.py --source data/raw/combined --out data/raw/combined_12k --n 12000 --seed 42

# 4) Counts, histograms, this document
python scripts/analyze_distribution.py
```

Outputs: `results/analysis/distribution.json`, `results/analysis/class_counts_*.png`,
`results/analysis/bbox_area_hist_*.png`, and this file.

## Domain notes (locked)

- **Combined** — industrial / construction mid-shot; 14 unified classes; the only train set.
- **Hard Hat Universe** — workplace helmet/head close-to-mid shots; held-out eval. `head`→`no_helmet` is an assumption.
- **Construction v28** — tiny overlapping slice (val **n=114**). Eval-only. `machinery`/`vehicle` stay here.
- **Do not merge** Construction into Combined. Construction already cloned images from Combined.

## Expected thin Combined classes

Expect `fall_detected`, `no_goggles`, and `gloves` / `no_gloves` to be thin. Flag them before claiming per-class SOTA.

## combined

Data not present at `C:\GitHub\Worker Safety PPE Detection Model\data\raw\combined`. Counts are pending a dataset download.

Domain: Combined is mostly construction / industrial mid-shot imagery with a 14-class PPE vocabulary (helmet/vest/goggles/gloves/mask plus `no_*`, person, cone, ladder, fall_detected).

## hardhat

Data not present at `C:\GitHub\Worker Safety PPE Detection Model\data\raw\hardhat`. Counts are pending a dataset download.

Domain: Hard Hat Universe is a workplace helmet/head domain (~7k). `head` → `no_helmet` is an eval assumption (implicit missing helmet) and must be scored separately so it does not pollute Combined metrics.

## construction

Data not present at `C:\GitHub\Worker Safety PPE Detection Model\data\raw\construction`. Counts are pending a dataset download.

Domain: Construction v28 is a tiny overlapping slice (val n=114, test n=82) cloned in part from Combined. Do not merge it back into Combined without perceptual hashing. `machinery` / `vehicle` are Construction-only.

## combined_12k

Data not present at `C:\GitHub\Worker Safety PPE Detection Model\data\raw\combined_12k`. Counts are pending a dataset download.

Domain: Stratified 12k subset of Combined used for the E0–E3 experiment grid.

## Inherited Construction plot (no raw images in this checkout)

The copied Ultralytics `results/labels.jpg` is from the **Construction** training set, not Combined:

- `Person` dominates instance count (on the order of ~9–10k boxes).
- `machinery` and `NO-Safety Vest` are next; `Mask` and `vehicle` are the thinnest.
- Many boxes are small (normalized width/height near 0). Spatial centers show a 4-quadrant mosaic pattern from training augs.

Treat those as qualitative. Exact Combined / HHU counts require the downloads above.
