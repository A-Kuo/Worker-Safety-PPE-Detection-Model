# Contributing

Contributions are licensed under AGPL-3.0-only, the same as the project.

## Setup

```bash
python -m pip install -e ".[cpu,dev]"        # runtime work: no torch needed
python -m pip install -e ".[cpu,gui,dev]"    # plus the live window
python -m pip install -r requirements.txt    # training / evaluation scripts (installs torch)
```

Use exactly one ONNX Runtime build per environment: `cpu` **or** `npu`, never both.

## Before opening a pull request

```bash
python -m pytest -q
python scripts/smoke_edge.py --require-npu
python -m build && python scripts/wheel_smoke.py
```

- The runtime (`src/ppe`, except `tools/export.py` and `inference.py`) must not import torch or
  ultralytics. A test checks this in a subprocess.
- Code that is part of the product goes in `src/ppe`; `scripts/` and `app/` are not in the wheel.
- New behaviour needs a test that does not require a dataset, a camera or an NPU. Use
  `ppe.testing.make_tiny_detector` for a model and the fakes in `tests/` for sessions and sources.

## Numbers

Any accuracy or latency figure in the docs must come from a committed result file that records the
model's SHA-256, and should state the sample size. Regenerate tables with
`python scripts/summarize_evidence.py`; do not type numbers by hand. Do not claim NPU execution without
a `ppe verify-npu` result, and do not compare devices on a single run.

## Third-party material

Do not commit model weights, dataset images, or media without a clear licence. Weights are release
assets (see `docs/RELEASING.md`); add new datasets and their licences to `ATTRIBUTION.md`.
