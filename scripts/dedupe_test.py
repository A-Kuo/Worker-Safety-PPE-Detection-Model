#!/usr/bin/env python3
"""Find test images that have a near-duplicate in train/valid, and write a clean test list.

Combined is assembled from many Roboflow sources, so the same photo (or a lightly re-cropped /
re-encoded copy) can land in both train and test. Scores on such test images reward memorization,
not generalization - and fine-tuning on a subset of train inflates them further. This uses a 64-bit
difference hash (dHash) and reports images whose nearest train/valid image is within ``--max-bits``.

  python scripts/dedupe_test.py                                   # whole test split
  python scripts/dedupe_test.py --subset data/raw/vest_annotated/test.txt --out-list ... --out-yaml ...

Writes a JSON report and, with --out-list/--out-yaml, a dataset yaml whose test split contains only
images with NO near-duplicate in train/valid (evaluate with ``eval.py --data <yaml> --split test``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _common import REPO_ROOT, dump_yaml, iter_split_images, load_schema  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data" / "processed" / "combined")
    parser.add_argument("--against", nargs="+", default=["train", "valid"])
    parser.add_argument("--subset", type=Path, default=None, help="text file of test image paths (default: whole test split)")
    parser.add_argument("--max-bits", type=int, default=6, help="Hamming distance <= this counts as a near-duplicate")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "analysis" / "test_dup_report.json")
    parser.add_argument("--out-list", type=Path, default=None)
    parser.add_argument("--out-yaml", type=Path, default=None)
    return parser.parse_args()


def dhash(path: Path) -> int | None:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    small = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _hash_all(paths):
    hashes, kept = [], []
    for p in paths:
        h = dhash(Path(p))
        if h is not None:
            hashes.append(h)
            kept.append(str(p))
    return kept, np.array(hashes, dtype=np.uint64)


_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount(x: np.ndarray) -> np.ndarray:
    """Set bits per uint64. np.bitwise_count needs numpy >= 2; the fallback stays vectorized
    (a Python loop over ~40k x 4.4k hash pairs would take hours)."""
    x = np.ascontiguousarray(x, dtype=np.uint64)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    return _POP8[x.view(np.uint8).reshape(-1, 8)].sum(axis=1)


def main() -> int:
    args = _parse_args()
    root = args.root if args.root.is_absolute() else REPO_ROOT / args.root

    ref_paths = [p for split in args.against for p in iter_split_images(root / split / "images")]
    if args.subset:
        test_paths = [ln for ln in args.subset.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        test_paths = [str(p) for p in iter_split_images(root / "test" / "images")]

    print(f"hashing {len(ref_paths)} reference images ({'+'.join(args.against)}) ...", flush=True)
    _, ref = _hash_all(ref_paths)
    print(f"hashing {len(test_paths)} test images ...", flush=True)
    kept, test = _hash_all(test_paths)

    nearest = np.array([int(_popcount(ref ^ h).min()) for h in test])
    is_dup = nearest <= args.max_bits
    clean = [p for p, dup in zip(kept, is_dup, strict=True) if not dup]
    report = {
        "reference_splits": args.against, "max_bits": args.max_bits,
        "test_images": len(kept), "exact_twins": int((nearest == 0).sum()),
        "near_duplicates": int(is_dup.sum()), "clean": len(clean),
        "near_duplicate_fraction": float(is_dup.mean()) if len(kept) else 0.0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if args.out_list:
        args.out_list.parent.mkdir(parents=True, exist_ok=True)
        args.out_list.write_text("\n".join(os.path.abspath(p).replace("\\", "/") for p in clean) + "\n", encoding="utf-8")
        if args.out_yaml:
            names = list(load_schema().UNIFIED_CLASS_NAMES)
            base = os.path.abspath(args.out_list.parent)
            dump_yaml(args.out_yaml, {"path": base, "train": args.out_list.name, "val": args.out_list.name,
                                      "test": args.out_list.name, "nc": len(names), "names": names})
            print(f"wrote {args.out_list} and {args.out_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
