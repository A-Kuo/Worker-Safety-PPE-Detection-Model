#!/usr/bin/env python3
"""Rewrite YOLO label class ids onto the unified Combined schema.

Walks a YOLO dataset, remaps each label line via ``ppe.schema``, and
writes a unified ``data.yaml``. Construction is **never** merged into Combined.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _common import (  # noqa: E402
    REPO_ROOT,
    apply_remap_line,
    build_id_map,
    discover_split_dirs,
    infer_mapping_kind,
    iter_split_images,
    label_path_for_image,
    link_or_copy,
    load_schema,
    lookup_mapping,
    mapping_for_source,
    read_dataset_names,
    write_unified_yaml,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="YOLO dataset root (contains data.yaml + splits)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination dataset root (images linked/copied, labels rewritten)",
    )
    parser.add_argument(
        "--mapping",
        choices=["combined", "construction", "hhu", "auto"],
        default="auto",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Rejected on purpose. Construction must not be merged into Combined.",
    )
    parser.add_argument(
        "--copy-images", action="store_true", help="Copy images instead of symlink/hardlink."
    )
    return parser.parse_args()


def _find_data_yaml(source: Path) -> Path:
    direct = source / "data.yaml"
    if direct.exists():
        return direct
    matches = list(source.rglob("data.yaml"))
    if not matches:
        raise FileNotFoundError(f"No data.yaml under {source}")
    matches.sort(key=lambda p: len(p.parts))
    return matches[0]


def _refuse_merge(mapping: str, source: Path, dest: Path) -> None:
    dest_s = dest.resolve().as_posix().lower()
    src_s = source.resolve().as_posix().lower()
    dest_is_combined = "combined" in dest_s and "construction" not in dest_s
    src_is_construction = mapping == "construction" or "construction" in src_s
    if src_is_construction and dest_is_combined:
        raise SystemExit(
            "Refusing to write Construction labels into a Combined destination. "
            "Construction is not merged into Combined. Doing that safely needs "
            "perceptual-hash deduplication, which is out of scope here."
        )


def main() -> int:
    args = _parse_args()
    if args.merge:
        raise SystemExit(
            "--merge is not supported. Construction stays held-out / eval-only. "
            "Never stitch Construction into Combined."
        )

    schema = load_schema()
    source = args.source if args.source.is_absolute() else REPO_ROOT / args.source
    dest = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    yaml_path = _find_data_yaml(source)
    dataset_root = yaml_path.parent
    names = read_dataset_names(yaml_path)
    kind = infer_mapping_kind(names) if args.mapping == "auto" else args.mapping
    _refuse_merge(kind, source, dest)

    name_mapping = mapping_for_source(schema, kind)
    # Key the mapping by the source yaml's actual class strings (case-insensitive match).
    effective = dict(name_mapping)
    for raw in names:
        unified = lookup_mapping(name_mapping, raw)
        if unified:
            effective[raw] = unified
    name_mapping = effective
    id_map = build_id_map(schema, names, name_mapping)
    dropped = [n for i, n in enumerate(names) if i not in id_map]

    splits = discover_split_dirs(dataset_root)
    if not splits:
        raise SystemExit(f"No train/valid/test image folders under {dataset_root}")

    dest.mkdir(parents=True, exist_ok=True)
    stats = {
        "mapping": kind,
        "source": str(source),
        "dropped_classes": dropped,
        "splits": {},
        "id_map": {str(k): v for k, v in id_map.items()},
    }

    for split, images_dir in splits.items():
        n_img = n_lab = n_drop = 0
        for image in iter_split_images(images_dir):
            rel = image.relative_to(images_dir)
            dest_img = dest / split / "images" / rel
            dest_lab = dest / split / "labels" / rel.with_suffix(".txt")
            if args.copy_images:
                import shutil

                dest_img.parent.mkdir(parents=True, exist_ok=True)
                if not dest_img.exists():
                    shutil.copy2(image, dest_img)
            else:
                link_or_copy(image, dest_img)
            n_img += 1

            src_lab = label_path_for_image(image)
            dest_lab.parent.mkdir(parents=True, exist_ok=True)
            out_lines: list[str] = []
            if src_lab.exists():
                for raw in src_lab.read_text(encoding="utf-8").splitlines():
                    if not raw.strip():
                        continue
                    remapped = apply_remap_line(schema, raw, names, name_mapping)
                    if remapped is None or not str(remapped).strip():
                        n_drop += 1
                        continue
                    out_lines.append(str(remapped).strip())
            dest_lab.write_text(
                ("\n".join(out_lines) + ("\n" if out_lines else "")), encoding="utf-8"
            )
            n_lab += 1
        stats["splits"][split] = {"images": n_img, "label_files": n_lab, "dropped_boxes": n_drop}
        print(f"{split}: {n_img} images, dropped {n_drop} unmapped boxes")

    train_rel = "train/images" if "train" in splits else next(iter(splits.values())).as_posix()
    val_key = (
        "valid" if "valid" in splits else ("train" if "train" in splits else next(iter(splits)))
    )
    val_rel = f"{val_key}/images"
    test_rel = "test/images" if "test" in splits else None
    out_yaml = dest / "data.yaml"
    write_unified_yaml(
        schema,
        out_yaml,
        train=train_rel,
        val=val_rel,
        test=test_rel,
        names=list(schema.UNIFIED_CLASS_NAMES),
    )

    (dest / "remap_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote remapped dataset to {dest}")
    if dropped:
        print(f"Dropped unmapped source classes: {dropped}")
    if kind == "hhu":
        print("NOTE: HHU `head` -> `no_helmet` is an assumption (implicit missing helmet).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
