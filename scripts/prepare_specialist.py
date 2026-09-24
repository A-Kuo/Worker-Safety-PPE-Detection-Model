#!/usr/bin/env python3
"""One command that prepares a specialist's dataset from scratch (download -> remap -> subset -> dedupe -> build).

    python scripts/prepare_specialist.py --exp helmet_specialist

Kept as a single script (rather than notebook cells) because the Kaggle notebook is Kaggle's own copy
and cannot be updated from GitHub, while scripts/ is re-cloned every run: adding a specialist then
needs no notebook change beyond ``EXP = "<name>"``.

Steps mirror the vest specialist's proven notebook path: every step reads the REMAPPED data
(data/processed/...), never data/raw labels. ``--skip-download`` reuses already-downloaded/remapped data.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _common import REPO_ROOT  # noqa: E402
from specialists import SPECIALISTS  # noqa: E402

# Early gate: fail here, not after training. Vest reference (local): train 3253 / valid 834 / test 145.
MIN_IMAGES = {"train": 500, "valid": 100, "test": 30}


def build_steps(exp: str, skip_download: bool = False) -> list[list[str]]:
    spec = SPECIALISTS[exp]
    py = sys.executable
    annotated = f"data/raw/{spec.annotated_dir}"
    steps: list[list[str]] = []
    if not skip_download:
        steps.append([py, "scripts/download_datasets.py", "--execute", "--only", "combined",
                      *([spec.external] if spec.external else [])])
    steps.append([py, "scripts/remap_labels.py", "--source", "data/raw/combined", "--out",
                  "data/processed/combined", "--mapping", "combined"])
    if spec.external:
        steps.append([py, "scripts/remap_labels.py", "--source", f"data/raw/{spec.external}", "--out",
                      f"data/processed/{spec.external}", "--mapping", spec.external])
    steps.append([py, "scripts/make_class_subset.py", "--classes", *spec.classes, "--out", annotated])
    # Drop test images that near-duplicate a train/valid image, so the test score can't be memorization.
    steps.append([py, "scripts/dedupe_test.py", "--subset", f"{annotated}/test.txt", "--out-list",
                  f"{annotated}/test_clean.txt", "--out-yaml", f"{annotated}/data_clean.yaml"])
    steps.append([py, "scripts/make_specialist_data.py", "--exp", exp])
    return steps


def check_counts(exp: str, root: Path = REPO_ROOT, min_scale: float = 1.0) -> dict[str, int]:
    counts = {s: len(list((root / "data" / "processed" / exp / s / "images").glob("*")))
              for s in ("train", "valid", "test")}
    print(f"{exp} images per split: {counts}")
    floor = {s: max(1, int(n * min_scale)) for s, n in MIN_IMAGES.items()}
    if any(counts[s] < floor[s] for s in counts):
        raise AssertionError(f"{exp} dataset looks wrong (below {floor}): {counts}")
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", choices=sorted(SPECIALISTS), required=True)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--min-scale", type=float, default=1.0,
                        help="scale the split-size floor (only for tiny synthetic rehearsals)")
    parser.add_argument("--dry-run", action="store_true", help="print the steps and exit")
    args = parser.parse_args(argv)
    steps = build_steps(args.exp, args.skip_download)
    for step in steps:
        print("+ " + " ".join(step[1:] if step[0] == sys.executable else step), flush=True)
        if not args.dry_run:
            subprocess.run(step, cwd=REPO_ROOT, check=True)
    if not args.dry_run:
        check_counts(args.exp, min_scale=args.min_scale)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
