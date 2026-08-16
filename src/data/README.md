# src/data

Dataset label-remapping and dedup utilities for Milestone M2
(`docs/PROJECT_PLAN.md`).

- `label_schema.py` — the unified PPE taxonomy and every source dataset's
  confirmed (or best-available) class mapping. Raises loudly on any
  unrecognized source class rather than silently dropping labels.
- `dedup.py` — perceptual-hash (average-hash) near-duplicate image detection,
  used to catch cross-source (and, as it turned out, within-source) image
  overlap before merging splits.

Both are unit-tested (`tests/test_label_schema.py`, `tests/test_dedup.py`) and
have been run end-to-end against real data via
`scripts/build_unified_dataset.py` — see `docs/DATA_DISTRIBUTION.md` for the
real output of that run.
