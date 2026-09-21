#!/usr/bin/env python3
"""Subset the remapped Combined data to images that annotate given classes.

Combined merges sources that each annotated only some of the 14 classes, so "no box for class X"
usually means "X was not annotated", not "X is absent" (e.g. 84% of test images with workers have
no vest/no_vest label at all). Measuring - or training - a class on images where nothing about it
was annotated punishes correct detections as false positives and teaches the model to suppress the
class. This selects only images where the class of interest WAS annotated:

  python scripts/make_class_subset.py --classes vest no_vest --out data/raw/vest_annotated

writes ``<out>/{train,valid,test}.txt`` (absolute image paths; labels are found by Ultralytics'
/images/ -> /labels/ swap, so they stay the remapped labels) and ``<out>/data.yaml``.
Uses abspath, never resolve(): symlinked processed images must not lead back to raw/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _common import REPO_ROOT, dump_yaml, iter_split_images, label_path_for_image, load_schema  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=REPO_ROOT / "data" / "processed" / "combined")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", required=True, help="keep images containing ANY of these unified classes")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    schema = load_schema()
    unified = list(schema.UNIFIED_CLASS_NAMES)
    unknown = [c for c in args.classes if c not in unified]
    if unknown:
        raise SystemExit(f"unknown classes {unknown}; choose from {unified}")
    wanted = {unified.index(c) for c in args.classes}

    source = args.source if args.source.is_absolute() else REPO_ROOT / args.source
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    root_abs = os.path.normcase(os.path.abspath(source))
    out.mkdir(parents=True, exist_ok=True)

    manifest = {"source": os.path.abspath(source), "classes": args.classes, "splits": {}}
    written = {}
    for split in ("train", "valid", "test"):
        images_dir = source / split / "images"
        if not images_dir.is_dir():
            continue
        kept, box_counts = [], {c: 0 for c in args.classes}
        total = 0
        for image in iter_split_images(images_dir):
            total += 1
            label = label_path_for_image(image)
            ids = []
            if label.exists():
                ids = [int(line.split()[0]) for line in label.read_text(encoding="utf-8").splitlines() if line.split()]
            if wanted.intersection(ids):
                line = os.path.abspath(image).replace("\\", "/")
                lbl_abs = os.path.normcase(os.path.abspath(label_path_for_image(Path(line))))
                if not lbl_abs.startswith(root_abs + os.sep):
                    raise SystemExit(f"{line} maps to a label outside {root_abs}; refusing (symlink resolved to raw?)")
                kept.append(line)
                for c in args.classes:
                    box_counts[c] += ids.count(unified.index(c))
        (out / f"{split}.txt").write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        written[split] = f"{split}.txt"
        manifest["splits"][split] = {"images": len(kept), "of": total, "boxes": box_counts}
        print(f"{split}: {len(kept)} / {total} images  boxes={box_counts}")

    payload = {"path": os.path.abspath(out), "train": written["train"], "val": written.get("valid", written["train"]),
               "nc": len(unified), "names": unified}
    if "test" in written:
        payload["test"] = written["test"]
    dump_yaml(out / "data.yaml", payload)
    (out / "subset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
