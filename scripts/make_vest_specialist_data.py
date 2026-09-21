#!/usr/bin/env python3
"""Build the 2-class (vest, no_vest) dataset for the vest specialist.

Why a separate model: Combined merges sources that each annotated only some classes, so 84% of
worker images have no vest status labeled. Training the 14-class model on that treats unlabeled
vests / plain torsos as background and suppresses ``no_vest``; fine-tuning that model on the few
vest-annotated images instead makes it forget every class *not* labeled there (helmet 0.82 -> 0.00
in a CPU proof of concept). A specialist trained ONLY on images where vest status was annotated has
nothing else to forget, and its two classes replace the main model's at inference.

Inputs (all already in the repo's pipeline):
  * data/raw/vest_annotated/{train,valid}.txt     from scripts/make_class_subset.py
  * data/raw/vest_annotated/test_clean.txt         from scripts/dedupe_test.py (test minus near-dups of train/valid)
  * data/processed/gap_vest/{train,valid}          novest/no-vest-detect, remapped (vest status fully annotated)

Output: data/processed/vest_specialist/{train,valid,test}/{images,labels} + data.yaml (nc=2).
Only vest (unified 2 -> 0) and no_vest (3 -> 1) boxes are kept; images without any are skipped.
External images that are near-duplicates of ANY Combined test image are dropped so they can't leak.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _common import REPO_ROOT, dump_yaml, iter_split_images, label_path_for_image, link_or_copy  # noqa: E402

UNIFIED_TO_SPECIALIST = {2: 0, 3: 1}  # vest, no_vest
SPECIALIST_NAMES = ["vest", "no_vest"]


def remap_label_lines(lines) -> list[str]:
    """Keep only vest/no_vest boxes and renumber them 0/1; everything else is dropped."""
    out = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
        except ValueError:
            continue
        if cls in UNIFIED_TO_SPECIALIST:
            out.append(" ".join([str(UNIFIED_TO_SPECIALIST[cls]), *parts[1:]]))
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vest-annotated", type=Path, default=REPO_ROOT / "data" / "raw" / "vest_annotated")
    parser.add_argument("--gap", type=Path, default=REPO_ROOT / "data" / "processed" / "gap_vest")
    parser.add_argument("--combined", type=Path, default=REPO_ROOT / "data" / "processed" / "combined")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "processed" / "vest_specialist")
    parser.add_argument("--no-gap", action="store_true", help="Combined-only (skip the external vest dataset)")
    parser.add_argument("--max-bits", type=int, default=6)
    return parser.parse_args()


def _entries(source_list: Path) -> list[Path]:
    return [Path(p) for p in source_list.read_text(encoding="utf-8").splitlines() if p.strip()]


def main() -> int:
    args = _parse_args()
    out = args.out
    plan: dict[str, list[tuple[str, Path]]] = {"train": [], "valid": [], "test": []}

    for split, listing in (("train", "train.txt"), ("valid", "valid.txt"), ("test", "test_clean.txt")):
        path = args.vest_annotated / listing
        if not path.is_file():
            raise SystemExit(f"{path} is missing - run make_class_subset.py / dedupe_test.py first.")
        plan[split] += [("cmb", p) for p in _entries(path)]

    if not args.no_gap:
        import numpy as np

        from dedupe_test import _hash_all, _popcount

        test_paths = [str(p) for p in iter_split_images(args.combined / "test" / "images")]
        print(f"hashing {len(test_paths)} Combined test images to keep external data out of them ...", flush=True)
        _, test_hashes = _hash_all(test_paths)
        dropped = 0
        for split in ("train", "valid"):
            images_dir = args.gap / split / "images"
            if not images_dir.is_dir():
                continue
            kept, hashes = _hash_all(iter_split_images(images_dir))
            for path, h in zip(kept, hashes, strict=True):
                if int(_popcount(test_hashes ^ np.uint64(h)).min()) <= args.max_bits:
                    dropped += 1
                else:
                    plan[split].append(("gap", Path(path)))
        print(f"dropped {dropped} external images that near-duplicate a Combined test image")

    stats = {}
    for split, items in plan.items():
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        n_img = n_vest = n_no = 0
        for prefix, image in items:
            label = label_path_for_image(image)
            if not label.is_file():
                continue
            lines = remap_label_lines(label.read_text(encoding="utf-8").splitlines())
            if not lines:
                continue
            stem = f"{prefix}_{image.stem}"
            link_or_copy(image, out / split / "images" / f"{stem}{image.suffix}")
            (out / split / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            n_img += 1
            n_vest += sum(1 for ln in lines if ln.startswith("0 "))
            n_no += sum(1 for ln in lines if ln.startswith("1 "))
        stats[split] = (n_img, n_vest, n_no)
        print(f"{split}: {n_img} images, vest={n_vest}, no_vest={n_no}")

    dump_yaml(out / "data.yaml", {"path": str(out.resolve()), "train": "train/images", "val": "valid/images",
                                  "test": "test/images", "nc": 2, "names": SPECIALIST_NAMES})
    print(f"Wrote {out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
