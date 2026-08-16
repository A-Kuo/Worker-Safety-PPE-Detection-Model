#!/usr/bin/env python3
"""Merge confirmed PPE dataset sources into data/unified/ under the schema in
src/data/label_schema.py.

Usage:
    python scripts/build_unified_dataset.py --source construction_site_safety=/path/to/root \\
        --source ultralytics_construction_ppe=/path/to/construction-ppe \\
        --out data/unified

Only sources with real, locally-available images can be merged - see
src/data/label_schema.py `SOURCES[...].status`. Currently that's just
`ultralytics_construction_ppe` (downloaded directly, no API key needed) and,
once you have Kaggle/Roboflow credentials, `construction_site_safety`.
`ppe_combined_model` and `hard_hat_universe` remain "pending_export" until we
have a Roboflow API key (see docs/PROJECT_PLAN.md Open Questions).

What this script does, end to end:
  1. For each `--source key=root_dir`, detect the source's own directory layout
     (Ultralytics-style `images/{split}` + `labels/{split}`, or Roboflow-style
     `{split}/images` + `{split}/labels`, with `val`/`valid` aliasing) and its
     own `data.yaml` class order.
  2. Remap every label file's class ids to the unified schema via
     `label_schema.remap_source_labels` (raises loudly on any unmapped class -
     never silently drops an unrecognized label).
  3. Copies (does not modify in place) each image + remapped label into
     `<out>/<split>/images` and `<out>/<split>/labels`, prefixing filenames with
     the source key to avoid collisions between sources.
  4. Runs perceptual-hash dedup (src/data/dedup.py) within each split, across
     all sources combined, and writes a report of any duplicate clusters found
     - especially watch for clusters spanning two different sources, which is
     the confirmed leakage risk between construction_site_safety and
     ppe_combined_model documented in docs/DATASET_NOTES.md.
  5. Writes `<out>/data.yaml` (unified class list) and a per-source, per-split,
     per-class instance-count CSV for the data-distribution analysis
     (docs/DATA_DISTRIBUTION.md).

This does NOT delete duplicates automatically - it reports them so a human can
decide which copy to keep (dedup is provided by find_duplicate_groups()).
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import label_schema  # noqa: E402
from src.data.dedup import find_duplicate_groups  # noqa: E402

CANONICAL_SPLITS = ["train", "val", "test"]
SPLIT_ALIASES = {"val": ["val", "valid"]}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _resolve_split_dir(root: Path, split: str, *layouts: str) -> tuple[Path, Path] | None:
    """Try `images/{split}`+`labels/{split}` first, then `{split}/images`+`{split}/labels`,
    trying each split alias (e.g. val/valid) in turn. Returns (images_dir, labels_dir)
    for the first combination that exists, or None."""
    aliases = SPLIT_ALIASES.get(split, [split])
    for alias in aliases:
        candidates = [
            (root / "images" / alias, root / "labels" / alias),
            (root / alias / "images", root / alias / "labels"),
        ]
        for images_dir, labels_dir in candidates:
            if images_dir.is_dir() and labels_dir.is_dir():
                return images_dir, labels_dir
    return None


def _load_source_class_names(root: Path) -> list[str]:
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        candidates = list(root.glob("*.yaml")) + list(root.glob("*.yml"))
        if not candidates:
            raise FileNotFoundError(f"No data.yaml found under {root}")
        yaml_path = candidates[0]
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = data["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def merge_source(source_key: str, root: Path, out_root: Path, stats: Counter) -> list[Path]:
    class_names = _load_source_class_names(root)
    id_map = label_schema.remap_source_labels(source_key, class_names)
    copied_images: list[Path] = []

    for split in CANONICAL_SPLITS:
        resolved = _resolve_split_dir(root, split)
        if resolved is None:
            print(f"  [{source_key}] no '{split}' split found under {root}, skipping")
            continue
        images_dir, labels_dir = resolved

        out_images = out_root / split / "images"
        out_labels = out_root / split / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        n_images = 0
        for label_path in sorted(labels_dir.glob("*.txt")):
            image_path = next(
                (images_dir / (label_path.stem + ext) for ext in IMAGE_EXTS if (images_dir / (label_path.stem + ext)).exists()),
                None,
            )
            if image_path is None:
                continue

            remapped_lines = []
            with open(label_path) as f:
                for line in f:
                    parts = line.split()
                    if not parts:
                        continue
                    old_id = int(parts[0])
                    new_id = id_map[old_id]
                    if new_id is None:
                        continue
                    remapped_lines.append(" ".join([str(new_id)] + parts[1:]))
                    stats[(source_key, split, label_schema.UNIFIED_CLASSES[new_id])] += 1

            dest_stem = f"{source_key}__{image_path.stem}"
            dest_image = out_images / (dest_stem + image_path.suffix)
            dest_label = out_labels / (dest_stem + ".txt")
            shutil.copy2(image_path, dest_image)
            dest_label.write_text("\n".join(remapped_lines) + ("\n" if remapped_lines else ""))
            copied_images.append(dest_image)
            n_images += 1

        print(f"  [{source_key}] {split}: copied {n_images} images")

    return copied_images


def write_unified_yaml(out_root: Path) -> None:
    data = {
        "path": str(out_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: name for i, name in enumerate(label_schema.UNIFIED_CLASSES)},
    }
    with open(out_root / "data.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def write_distribution_csv(stats: Counter, out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "split", "unified_class", "instance_count"])
        for (source, split, cls), count in sorted(stats.items()):
            writer.writerow([source, split, cls, count])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="KEY=PATH",
        help="Repeatable. KEY must be a key in src/data/label_schema.SOURCES, PATH is the dataset root.",
    )
    parser.add_argument("--out", default="data/unified", help="Output directory (default: data/unified)")
    parser.add_argument(
        "--dedup-threshold",
        type=int,
        default=5,
        help="Max aHash Hamming distance to treat two images as duplicates (default: 5)",
    )
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    all_copied: list[Path] = []

    for entry in args.source:
        key, _, path = entry.partition("=")
        if key not in label_schema.SOURCES:
            raise SystemExit(f"Unknown source key '{key}'. Known: {list(label_schema.SOURCES)}")
        source = label_schema.SOURCES[key]
        print(f"Merging source '{key}' ({source.display_name}, status={source.status})")
        all_copied.extend(merge_source(key, Path(path), out_root, stats))

    write_unified_yaml(out_root)
    write_distribution_csv(stats, out_root / "distribution.csv")

    print(f"\nRunning dedup (threshold={args.dedup_threshold}) over {len(all_copied)} merged images...")
    groups = find_duplicate_groups(all_copied, threshold=args.dedup_threshold)
    if groups:
        report_path = out_root / "duplicate_report.txt"
        with open(report_path, "w") as f:
            for g in groups:
                f.write(f"[max_dist={g.max_internal_distance}]\n")
                for p in g.paths:
                    f.write(f"  {p}\n")
        print(f"Found {len(groups)} duplicate cluster(s) - see {report_path}")
    else:
        print("No duplicate clusters found at this threshold.")

    print(f"\nDone. Unified dataset at {out_root}, config at {out_root / 'data.yaml'}, "
          f"class distribution at {out_root / 'distribution.csv'}.")


if __name__ == "__main__":
    main()
