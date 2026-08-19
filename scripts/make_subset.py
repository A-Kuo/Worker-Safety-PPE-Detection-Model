#!/usr/bin/env python3
"""Build a stratified 12k subset of Combined (class-presence + split ratios).

Writes file lists and a yaml under ``data/raw/combined_12k/``. Does not copy
the 44k image tree. Never merges Construction into Combined.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _common import (  # noqa: E402
    REPO_ROOT,
    discover_split_dirs,
    dump_yaml,
    infer_mapping_kind,
    iter_split_images,
    label_path_for_image,
    load_schema,
    parse_yolo_label,
    read_dataset_names,
)

DEFAULT_SOURCE = REPO_ROOT / "data" / "raw" / "combined"
DEFAULT_OUT = REPO_ROOT / "data" / "raw" / "combined_12k"
DEFAULT_N = 12000
DEFAULT_SEED = 42


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--merge-construction",
        action="store_true",
        help="Rejected. Construction must not be mixed into the Combined subset.",
    )
    return parser.parse_args()


def _find_yaml(source: Path) -> Path:
    for cand in (source / "data.yaml", *source.rglob("data.yaml")):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No data.yaml under {source}")


def _records_for_split(images_dir: Path) -> list[dict]:
    recs = []
    for image in iter_split_images(images_dir):
        boxes = parse_yolo_label(label_path_for_image(image))
        classes = {int(row[0]) for row in boxes if row}
        recs.append({"image": image, "classes": classes})
    return recs


def stratified_sample(recs: list[dict], n: int, seed: int) -> list[dict]:
    """Preserve class-presence proportions; guarantee rare classes appear if possible."""
    rng = random.Random(seed)
    n = min(n, len(recs))
    if n >= len(recs):
        return list(recs)

    class_to_idxs: dict[int, list[int]] = defaultdict(list)
    for i, rec in enumerate(recs):
        for cls in rec["classes"]:
            class_to_idxs[cls].append(i)

    total = len(recs)
    targets = {c: n * (len(idxs) / total) for c, idxs in class_to_idxs.items()}
    selected: set[int] = set()
    counts: Counter[int] = Counter()

    for cls, idxs in sorted(class_to_idxs.items(), key=lambda kv: len(kv[1])):
        if not any(cls in recs[i]["classes"] for i in selected) and idxs:
            pick = rng.choice(idxs)
            selected.add(pick)
            counts.update(recs[pick]["classes"])

    remaining = [i for i in range(total) if i not in selected]
    rng.shuffle(remaining)

    def deficit(idx: int) -> float:
        return sum(max(0.0, targets[c] - counts[c]) for c in recs[idx]["classes"])

    while len(selected) < n and remaining:
        pool = remaining if len(remaining) <= 2500 else rng.sample(remaining, 2500)
        pool.sort(key=deficit, reverse=True)
        pick = pool[0] if deficit(pool[0]) > 0 else remaining[0]
        selected.add(pick)
        counts.update(recs[pick]["classes"])
        remaining.remove(pick)

    return [recs[i] for i in sorted(selected)]


def main() -> int:
    args = _parse_args()
    if args.merge_construction:
        raise SystemExit("Refusing to merge Construction into the Combined 12k subset.")

    source = args.source if args.source.is_absolute() else REPO_ROOT / args.source
    dest = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if not source.exists():
        raise SystemExit(
            f"Combined source not found: {source}. "
            "Download with scripts/download_datasets.py --execute (requires ROBOFLOW_API_KEY)."
        )
    if "construction" in source.resolve().as_posix().lower():
        raise SystemExit("Source looks like Construction. Combined subset only - no merge.")

    yaml_path = _find_yaml(source)
    dataset_root = yaml_path.parent
    names = read_dataset_names(yaml_path)
    if infer_mapping_kind(names) == "construction":
        raise SystemExit("Source yaml looks like Construction. Will not subset it into Combined.")

    splits = discover_split_dirs(dataset_root)
    if not splits:
        raise SystemExit(f"No image splits under {dataset_root}")

    split_recs = {split: _records_for_split(images_dir) for split, images_dir in splits.items()}
    total = sum(len(v) for v in split_recs.values())
    if total == 0:
        raise SystemExit(f"No images found under {dataset_root}")

    target_n = min(args.n, total)
    chosen: dict[str, list[dict]] = {}
    allocated = 0
    ordered = [s for s in ("train", "valid", "test") if s in split_recs]
    for i, split in enumerate(ordered):
        recs = split_recs[split]
        if i == len(ordered) - 1:
            take = target_n - allocated
        else:
            take = round(target_n * (len(recs) / total))
        take = max(0, min(len(recs), take))
        chosen[split] = stratified_sample(recs, take, args.seed + i)
        allocated += len(chosen[split])

    dest.mkdir(parents=True, exist_ok=True)
    list_rel: dict[str, str] = {}
    presence_before: Counter[int] = Counter()
    presence_after: Counter[int] = Counter()
    for recs in split_recs.values():
        for rec in recs:
            presence_before.update(rec["classes"])
    for split, recs in chosen.items():
        list_name = f"{split}.txt"
        list_path = dest / list_name
        lines = [str(rec["image"].resolve()).replace("\\", "/") for rec in recs]
        list_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        list_rel[split] = list_name
        for rec in recs:
            presence_after.update(rec["classes"])
        print(f"{split}: {len(recs)} / {len(split_recs[split])}")

    names_out = list(names)
    try:
        schema = load_schema()
        if infer_mapping_kind(names) == "combined" and set(n.casefold() for n in names) == set(
            n.casefold() for n in schema.UNIFIED_CLASS_NAMES
        ):
            names_out = list(schema.UNIFIED_CLASS_NAMES)
    except ImportError:
        pass

    payload = {
        "path": str(dest.resolve()),
        "train": list_rel.get("train", list_rel.get("valid", next(iter(list_rel.values())))),
        "val": list_rel.get("valid", list_rel.get("train")),
        "nc": len(names_out),
        "names": names_out,
    }
    if "test" in list_rel:
        payload["test"] = list_rel["test"]
    dump_yaml(dest / "data.yaml", payload)

    def _named(counter: Counter[int]) -> dict[str, int]:
        return {names[k] if k < len(names) else str(k): int(v) for k, v in sorted(counter.items())}

    manifest = {
        "source": str(source.resolve()),
        "n_requested": args.n,
        "n_selected": sum(len(v) for v in chosen.values()),
        "seed": args.seed,
        "split_counts": {k: len(v) for k, v in chosen.items()},
        "class_presence_full": _named(presence_before),
        "class_presence_subset": _named(presence_after),
        "note": (
            "Stratified on class presence (images containing the class) and on "
            "split proportions. Construction is not part of this subset."
        ),
    }
    (dest / "subset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {dest / 'data.yaml'} ({manifest['n_selected']} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
