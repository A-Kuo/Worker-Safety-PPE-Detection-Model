# Data distribution

Run before any new Combined training. It covers class imbalance, split
sizes, box scale, and the domain gap between Combined, HHU, and the small
Construction val split.

## How to regenerate

```bash
# 1) Download (requires ROBOFLOW_API_KEY; omit --execute for a dry-run)
python scripts/download_datasets.py --dry-run
python scripts/download_datasets.py --execute

# 2) Remap onto the unified schema (never merge Construction into Combined)
python scripts/remap_labels.py --source data/raw/combined \
    --out data/processed/combined --mapping combined
python scripts/remap_labels.py --source data/raw/hardhat \
    --out data/processed/hardhat --mapping hhu
python scripts/remap_labels.py --source data/raw/construction \
    --out data/processed/construction --mapping construction

# 3) Optional 12k grid subset
python scripts/make_subset.py --source data/raw/combined \
    --out data/raw/combined_12k --n 12000 --seed 42

# 4) Counts, histograms, this document
python scripts/analyze_distribution.py
```

Outputs: `results/analysis/distribution.json`, `results/analysis/class_counts_*.png`,
`results/analysis/bbox_area_hist_*.png`, and this file.

## Domain notes

- Combined: industrial and construction mid-shot, 14 unified classes,
  the only training set.
- Hard Hat Universe: workplace helmet and head shots, held out for eval.
  Reading `head` as `no_helmet` is an assumption.
- Construction v28: small overlapping slice (val n=114), eval only.
  `machinery` and `vehicle` stay here.
- Construction is not merged into Combined; it already clones images from it.

## Expected thin Combined classes

`fall_detected`, `no_goggles`, `gloves`, and `no_gloves` are all thin,
so their per-class numbers carry little weight.

## combined

Data not present at `/home/user/Worker-Safety-PPE-Detection-Model/data/raw/combined`. Counts are pending a dataset download.

Domain: Combined is mostly construction and industrial mid-shot imagery, on a 14-class vocabulary: helmet, vest, goggles, gloves, mask and their `no_*` counterparts, plus person, cone, ladder, fall_detected.

## hardhat

Data not present at `/home/user/Worker-Safety-PPE-Detection-Model/data/raw/hardhat`. Counts are pending a dataset download.

Domain: Hard Hat Universe is a workplace helmet and head domain of roughly 7k images. Reading `head` as `no_helmet` is an assumption, so it is scored separately and kept out of Combined metrics.

## construction

Data not present at `/home/user/Worker-Safety-PPE-Detection-Model/data/raw/construction`. Counts are pending a dataset download.

Domain: Construction v28 is a small overlapping slice (val n=114, test n=82), partly cloned from Combined. Merging it back needs perceptual hashing first. `machinery` and `vehicle` are Construction-only.

## combined_12k

Data not present at `/home/user/Worker-Safety-PPE-Detection-Model/data/raw/combined_12k`. Counts are pending a dataset download.

Domain: Stratified 12k subset of Combined used for the E0-E3 experiment grid.

## Inherited Construction plot (no raw images in this checkout)

The inherited `results/labels.jpg` covers the Construction training
set, not Combined:

- `Person` dominates the instance count, on the order of 9-10k boxes.
- `machinery` and `NO-Safety Vest` are next; `Mask` and `vehicle` are the thinnest.
- Most boxes are small (normalized width and height near 0). Centers
  fall into a four-quadrant pattern left over from mosaic augmentation.

Treat those as qualitative. Exact Combined / HHU counts require the downloads above.
