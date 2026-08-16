# Data Distribution Analysis — Milestone M2

Status: **Merge pipeline built and smoke-tested end-to-end against one real,
confirmed source.** Two of the four candidate sources remain blocked on a
Roboflow API key (see `PROJECT_PLAN.md` Open Questions) and are not yet
reflected in the numbers below.

## 1. What was actually run

`scripts/build_unified_dataset.py` was run for real against the downloaded
**Ultralytics Construction-PPE** dataset (1,416 images, no API key required —
see `DATASET_NOTES.md` §3). This is a genuine end-to-end execution of the
pipeline described in `PROJECT_PLAN.md` M2 — label remapping, cross-source dedup
scanning, unified `data.yaml` generation, and a per-class distribution CSV — not
a plan or a mock. Output artifacts are committed under
`docs/assets/m2_smoke_test/` for reference:

- `unified_data.yaml` — the generated unified-schema config for this run.
- `distribution.csv` — per-source/per-split/per-class instance counts.
- `duplicate_report.txt` — every near-duplicate cluster the dedup step found.

The other three sources (`construction_site_safety`, `ppe_combined_model`,
`hard_hat_universe`) are wired into `src/data/label_schema.py` with confirmed
or best-available class mappings, but weren't merged here because their actual
images aren't accessible without credentials (see `BASELINE_METRICS.md` §4 and
`PROJECT_PLAN.md` Open Question #3). Re-run the same script once those are
available — no code changes needed, just `--source construction_site_safety=...`
etc.

## 2. Per-class distribution (Ultralytics Construction-PPE only, so far)

From `docs/assets/m2_smoke_test/distribution.csv`, summed across train+val+test:

| Unified class | Total instances | Train | Val | Test |
|---|---|---|---|---|
| person | 2,245 | 1,770 | 239 | 236 |
| helmet | 1,734 | 1,341 | 201 | 192 |
| vest | 1,618 | 1,269 | 171 | 178 |
| boots | 1,597 | 1,235 | 151 | 211 |
| gloves | 1,445 | 1,146 | 136 | 163 |
| no_vest | 797 | 651 | 81 | 65 |
| goggles | 518 | 419 | 47 | 52 |
| no_gloves | 556 | 442 | 56 | 58 |
| no_helmet | 485 | 400 | 45 | 40 |
| no_goggles | 411 | 337 | 41 | 33 |
| no_boots | 115 | 88 | 4 | 23 |
| *(machinery/vehicle/cone/ladder/mask/no_mask)* | 0 | 0 | 0 | 0 |

Observations:

- This source alone is far better balanced across the "worn PPE" classes
  (`helmet`/`vest`/`boots`/`gloves` all in the 1,400–1,750 range) than the base
  Construction Site Safety dataset was (recall `BASELINE_METRICS.md`: base
  dataset's `Person` ≈ 9,550 vs. `Mask`/`vehicle` ≈ 1,550 — a ~6× imbalance).
  Merging the two should meaningfully improve overall class balance for
  `helmet`/`vest`, at the cost of reintroducing the base dataset's `Person`
  dominance once it's added.
- `no_boots` is the clear minority negative class here (115 total, 88 of them
  in train) — the smallest cell in the whole table. This is a direct, measured
  imbalance signal for the focal-loss/threshold-tuning work in M3, not a guess.
- `machinery`, `vehicle`, `cone`, `ladder`, `mask`, `no_mask` are entirely
  absent from this source (it simply has no annotations for them) — these
  classes depend entirely on the base dataset (`mask`/`no_mask`/`cone`) and the
  Combined Model export (`ladder`, possibly `machinery`/`vehicle` — unconfirmed,
  see `DATASET_NOTES.md`). Until those are merged in, any model trained on just
  this source cannot detect those classes at all.

## 3. Real finding from the dedup step: cross-split near-duplicates, even within a single "clean" source

Running the dedup step (perceptual average-hash, threshold=5 out of a max
possible 64) over this single dataset's 1,416 images found **122 duplicate
clusters**, and **57 of those 122 (47%) contain images from more than one of
train/val/test**. Cluster sizes ranged from 2 to 42 images; several large
clusters have consecutively-numbered filenames (e.g. `image850`, `image851`,
`image854`, `image855`... alongside `image1002`, `image1004`, `image1011`...),
which is the classic signature of a dataset built by extracting frames from a
handful of source videos — nearly-identical consecutive frames scattered
randomly across train/val/test by whatever split logic was used upstream.

**This means Ultralytics' own official Construction-PPE dataset most likely has
some degree of train/test leakage already**, independent of anything we're
merging in — its own published metrics may be somewhat optimistic for the
same reason the base dataset's numbers need scrutiny (`BASELINE_METRICS.md`).
Practical implications for this project:

1. When actually training on this dataset, consider re-splitting after dedup
   (assign each duplicate cluster entirely to one split, never splitting a
   cluster across train/val/test) rather than trusting the original split
   boundaries — this is a general fix, not specific to our merge.
2. This validates spending the effort on the dedup step at all — it caught a
   real problem in a dataset we didn't even construct ourselves, on the very
   first real run.
3. `threshold=5` is a starting point, not tuned against a hand-labeled set of
   true/false positive pairs — a full rigor pass should sample ~20 flagged
   pairs and ~20 unflagged near-neighbors and manually verify precision/recall
   of the dedup step itself before relying on it for the final merge.

## 4. Next steps once the Roboflow API key is available

1. Export `personal-protective-equipment-combined-model` and
   `hard-hat-universe-0dy7t`, confirm their real class lists against
   `src/data/label_schema.py` (currently `status="pending_export"` for both —
   update to `"confirmed"` once verified), and re-run
   `scripts/build_unified_dataset.py` with all sources included.
2. Specifically check for cross-source duplicate clusters between
   `construction_site_safety` and `ppe_combined_model` — this is the
   confirmed (not hypothetical) risk from `DATASET_NOTES.md` §"Confirmed ...
   cross-dataset image overlap risk".
3. Re-generate this document with the full 3–4-source distribution table and
   an explicit imbalance/domain-shift write-up per source, as originally
   scoped in `PROJECT_PLAN.md` M2.
