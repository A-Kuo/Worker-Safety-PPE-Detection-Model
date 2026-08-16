# scripts

- `build_unified_dataset.py` — Milestone M2's reproducible dataset-merge
  pipeline: remaps every configured source's labels to the unified taxonomy
  (`src/data/label_schema.py`), copies images+labels into one output tree,
  runs cross-source dedup (`src/data/dedup.py`), and writes a unified
  `data.yaml` plus a per-class distribution CSV. Implemented and smoke-tested
  end-to-end against one real source — see `docs/DATA_DISTRIBUTION.md` for the
  actual run's output. Run `python scripts/build_unified_dataset.py --help` for
  usage.
