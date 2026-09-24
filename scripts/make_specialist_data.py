#!/usr/bin/env python3
"""Build the 2-class dataset for a class-pair specialist (see scripts/specialists.py).

Why a separate model: Combined merges sources that each annotated only some classes, so most worker
images have no label for a given pair (84% carry no vest status). Training the 14-class model on that
treats unlabeled vests / helmets as background and suppresses the ``no_*`` class; fine-tuning it on the
few annotated images instead makes it forget every class not labeled there (helmet 0.82 -> 0.00 in a
CPU proof of concept). A specialist trained ONLY on images that annotate its pair has nothing else to
forget, and its classes replace the main model's at inference.

Inputs (all already in the repo's pipeline), for ``--exp vest_specialist`` (helmet analogous):
  * data/raw/<annotated_dir>/{train,valid}.txt   from scripts/make_class_subset.py
  * data/raw/<annotated_dir>/test_clean.txt      from scripts/dedupe_test.py (test minus near-dups of train/valid)
  * data/processed/<external>/{train,valid}      optional fully-annotated external data, remapped

Output: data/processed/<exp>/{train,valid,test}/{images,labels} + data.yaml (nc=2).
Only the pair's boxes are kept, renumbered 0/1; images without any are skipped. External images that
are near-duplicates of ANY Combined test image are dropped so they can't leak.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _common import REPO_ROOT, dump_yaml, iter_split_images, label_path_for_image, link_or_copy, load_schema  # noqa: E402
from specialists import SPECIALISTS, id_map  # noqa: E402


def remap_label_lines(lines, mapping: dict[int, int]) -> list[str]:
    """Keep only boxes whose class is in ``mapping`` and renumber them; everything else is dropped."""
    out = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
        except ValueError:
            continue
        if cls in mapping:
            out.append(" ".join([str(mapping[cls]), *parts[1:]]))
    return out


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", choices=sorted(SPECIALISTS), default="vest_specialist")
    parser.add_argument("--combined", type=Path, default=REPO_ROOT / "data" / "processed" / "combined")
    parser.add_argument("--no-external", "--no-gap", dest="no_external", action="store_true",
                        help="Combined-only (skip the specialist's external dataset)")
    parser.add_argument("--max-bits", type=int, default=6)
    return parser.parse_args(argv)


def _entries(source_list: Path) -> list[Path]:
    return [Path(p) for p in source_list.read_text(encoding="utf-8").splitlines() if p.strip()]


def main(argv=None) -> int:
    args = _parse_args(argv)
    spec = SPECIALISTS[args.exp]
    mapping = id_map(spec, load_schema().UNIFIED_CLASS_NAMES)
    annotated = REPO_ROOT / "data" / "raw" / spec.annotated_dir
    out = REPO_ROOT / "data" / "processed" / spec.exp
    plan: dict[str, list[tuple[str, Path]]] = {"train": [], "valid": [], "test": []}

    for split, listing in (("train", "train.txt"), ("valid", "valid.txt"), ("test", "test_clean.txt")):
        path = annotated / listing
        if not path.is_file():
            raise SystemExit(f"{path} is missing - run make_class_subset.py / dedupe_test.py first.")
        plan[split] += [("cmb", p) for p in _entries(path)]

    if spec.external and not args.no_external:
        import numpy as np

        from dedupe_test import _hash_all, _popcount

        external = REPO_ROOT / "data" / "processed" / spec.external
        test_paths = [str(p) for p in iter_split_images(args.combined / "test" / "images")]
        print(f"hashing {len(test_paths)} Combined test images to keep external data out of them ...", flush=True)
        _, test_hashes = _hash_all(test_paths)
        dropped = 0
        for split in ("train", "valid"):
            images_dir = external / split / "images"
            if not images_dir.is_dir():
                continue
            kept, hashes = _hash_all(iter_split_images(images_dir))
            for path, h in zip(kept, hashes, strict=True):
                if int(_popcount(test_hashes ^ np.uint64(h)).min()) <= args.max_bits:
                    dropped += 1
                else:
                    plan[split].append(("gap", Path(path)))
        print(f"dropped {dropped} external images that near-duplicate a Combined test image")

    for split, items in plan.items():
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        n_img, counts = 0, [0] * len(spec.classes)
        for prefix, image in items:
            label = label_path_for_image(image)
            if not label.is_file():
                continue
            lines = remap_label_lines(label.read_text(encoding="utf-8").splitlines(), mapping)
            if not lines:
                continue
            stem = f"{prefix}_{image.stem}"
            link_or_copy(image, out / split / "images" / f"{stem}{image.suffix}")
            (out / split / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            n_img += 1
            for i in range(len(spec.classes)):
                counts[i] += sum(1 for ln in lines if ln.startswith(f"{i} "))
        print(f"{split}: {n_img} images, " + ", ".join(f"{c}={n}" for c, n in zip(spec.classes, counts, strict=True)))

    dump_yaml(out / "data.yaml", {"path": str(out.resolve()), "train": "train/images", "val": "valid/images",
                                  "test": "test/images", "nc": len(spec.classes), "names": list(spec.classes)})
    print(f"Wrote {out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
