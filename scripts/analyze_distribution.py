#!/usr/bin/env python3
"""Class / split / bbox-area analysis for Combined, HHU, and Construction.

Writes ``results/analysis/`` plots (if matplotlib is available),
``results/analysis/distribution.json``, and refreshes ``docs/data_distribution.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections import Counter
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _common import (  # noqa: E402
    REPO_ROOT,
    discover_split_dirs,
    infer_mapping_kind,
    iter_split_images,
    label_path_for_image,
    parse_yolo_label,
    read_dataset_names,
)

ANALYSIS_DIR = REPO_ROOT / "results" / "analysis"
DOCS_PATH = REPO_ROOT / "docs" / "data_distribution.md"

DATASET_DEFAULTS = {
    "combined": REPO_ROOT / "data" / "raw" / "combined",
    "combined_12k": REPO_ROOT / "data" / "raw" / "combined_12k",
    "hardhat": REPO_ROOT / "data" / "raw" / "hardhat",
    "construction": REPO_ROOT / "data" / "raw" / "construction",
}

DOMAIN_NOTES = {
    "combined": (
        "Combined is mostly construction and industrial mid-shot imagery, on a "
        "14-class vocabulary: helmet, vest, goggles, gloves, mask and their `no_*` "
        "counterparts, plus person, cone, ladder, fall_detected."
    ),
    "hardhat": (
        "Hard Hat Universe is a workplace helmet and head domain of roughly 7k "
        "images. Reading `head` as `no_helmet` is an assumption, so it is scored "
        "separately and kept out of Combined metrics."
    ),
    "construction": (
        "Construction v28 is a small overlapping slice (val n=114, test n=82), "
        "partly cloned from Combined. Merging it back needs perceptual hashing "
        "first. `machinery` and `vehicle` are Construction-only."
    ),
    "combined_12k": "Stratified 12k subset of Combined used for the E0-E3 experiment grid.",
}

EXPECTED_THIN = ("fall_detected", "no_goggles", "gloves", "no_gloves")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", default=["combined", "hardhat", "construction", "combined_12k"]
    )
    parser.add_argument("--out-json", type=Path, default=ANALYSIS_DIR / "distribution.json")
    parser.add_argument("--out-md", type=Path, default=DOCS_PATH)
    return parser.parse_args()


def _find_yaml(root: Path) -> Path | None:
    for cand in (root / "data.yaml", *root.glob("**/data.yaml")):
        if cand.exists():
            return cand
    return None


def analyze_dataset(name: str, root: Path) -> dict:
    yaml_path = _find_yaml(root)
    dataset_root = yaml_path.parent if yaml_path else root
    names = read_dataset_names(yaml_path) if yaml_path else []
    splits = discover_split_dirs(dataset_root)

    # File-list yaml (12k subset): read txt lists.
    if yaml_path and not splits:
        payload = {}
        try:
            from _common import load_yaml

            payload = load_yaml(yaml_path)
        except Exception:  # noqa: BLE001
            payload = {}
        for split_key, field in (("train", "train"), ("valid", "val"), ("test", "test")):
            rel = payload.get(field)
            if not rel:
                continue
            list_path = Path(rel)
            if not list_path.is_absolute():
                list_path = Path(payload.get("path", dataset_root)) / rel
            if list_path.suffix == ".txt" and list_path.exists():
                images = [
                    Path(line.strip())
                    for line in list_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                splits[split_key] = images  # type: ignore[assignment]

    box_counts: dict[str, Counter[str]] = {}
    image_presence: dict[str, Counter[str]] = {}
    images_per_split: dict[str, int] = {}
    empty_images: dict[str, int] = {}
    areas: list[float] = []
    pos_neg = {"images_with_no_star": 0, "images_with_positive_ppe": 0, "empty_label_images": 0}
    no_star_tokens = {
        "no_helmet",
        "no_vest",
        "no_goggles",
        "no_gloves",
        "no_mask",
        "no-hardhat",
        "no-mask",
        "no-safety vest",
        "no-goggles",
        "no-gloves",
    }
    pos_tokens = {"helmet", "hardhat", "vest", "safety vest", "goggles", "gloves", "mask"}

    normalized_splits: dict[str, list[Path]] = {}
    for split, value in splits.items():
        if isinstance(value, list):
            normalized_splits[split] = value
        else:
            normalized_splits[split] = list(iter_split_images(value))

    for split, images in normalized_splits.items():
        bcount: Counter[str] = Counter()
        pcount: Counter[str] = Counter()
        empty = 0
        for image in images:
            boxes = parse_yolo_label(label_path_for_image(image))
            present: set[str] = set()
            if not boxes:
                empty += 1
                pos_neg["empty_label_images"] += 1
            for row in boxes:
                cls_id = int(row[0])
                label = names[cls_id] if cls_id < len(names) else str(cls_id)
                bcount[label] += 1
                present.add(label)
                if len(row) >= 5:
                    areas.append(max(0.0, float(row[3]) * float(row[4])))
            pcount.update(present)
            folded = {x.casefold() for x in present}
            if folded & no_star_tokens:
                pos_neg["images_with_no_star"] += 1
            if folded & pos_tokens:
                pos_neg["images_with_positive_ppe"] += 1
        box_counts[split] = bcount
        image_presence[split] = pcount
        images_per_split[split] = len(images)
        empty_images[split] = empty

    total_boxes = Counter()
    total_presence = Counter()
    for split in box_counts:
        total_boxes.update(box_counts[split])
        total_presence.update(image_presence[split])

    imbalance: list[str] = []
    if total_boxes:
        mx, mn = max(total_boxes.values()), min(total_boxes.values())
        if mn == 0 or mx / max(mn, 1) >= 10:
            imbalance.append(f"Box-count ratio max/min = {mx}/{mn} = {mx / max(mn, 1):.1f}x.")
        total = sum(total_boxes.values())
        for cls, n in total_boxes.items():
            if n / total < 0.01:
                imbalance.append(f"`{cls}` is <1% of boxes ({n}/{total}).")
        for token in EXPECTED_THIN:
            for cls in total_boxes:
                if token in cls.casefold().replace("-", "_").replace(" ", "_"):
                    imbalance.append(f"Thin class as expected: `{cls}` ({total_boxes[cls]} boxes).")

    area_hist = _histogram(areas, edges=(0.0, 0.01, 0.04, 0.09, 0.25, 1.01))
    return {
        "name": name,
        "root": str(root),
        "present": bool(normalized_splits),
        "kind": infer_mapping_kind(names) if names else None,
        "names": names,
        "images_per_split": images_per_split,
        "empty_label_images": empty_images,
        "box_counts": {s: dict(c) for s, c in box_counts.items()},
        "image_presence": {s: dict(c) for s, c in image_presence.items()},
        "box_counts_total": dict(total_boxes),
        "image_presence_total": dict(total_presence),
        "bbox_area_hist": area_hist,
        "bbox_area_n": len(areas),
        "pos_neg": pos_neg,
        "imbalance_flags": sorted(set(imbalance)),
        "domain_note": DOMAIN_NOTES.get(name, ""),
        "areas": areas,
    }


def _histogram(values: list[float], edges: tuple[float, ...]) -> list[dict]:
    buckets = []
    for lo, hi in zip(edges, edges[1:], strict=False):
        n = sum(1 for v in values if (v >= lo if lo == edges[0] else v > lo) and v <= hi)
        buckets.append({"lo": lo, "hi": hi, "count": n})
    return buckets


def _maybe_plots(reports: list[dict]) -> list[str]:
    written: list[str] = []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping PNG histograms.", file=sys.stderr)
        return written

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    for report in reports:
        if not report["present"]:
            continue
        names = report["names"] or list(report["box_counts_total"])
        totals = [report["box_counts_total"].get(n, 0) for n in names]
        if names:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(range(len(names)), totals)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha="right")
            ax.set_ylabel("boxes")
            ax.set_title(f"{report['name']} box counts")
            fig.tight_layout()
            path = ANALYSIS_DIR / f"class_counts_{report['name']}.png"
            fig.savefig(path, dpi=120)
            plt.close(fig)
            written.append(str(path))

        areas = report.get("areas") or []
        if areas:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(areas, bins=40, range=(0, min(1.0, max(areas) if areas else 1.0)))
            ax.set_xlabel("normalized bbox area (w*h)")
            ax.set_ylabel("count")
            ax.set_title(f"{report['name']} bbox area")
            fig.tight_layout()
            path = ANALYSIS_DIR / f"bbox_area_hist_{report['name']}.png"
            fig.savefig(path, dpi=120)
            plt.close(fig)
            written.append(str(path))
    return written


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_md(reports: list[dict], plot_paths: list[str]) -> str:
    lines = [
        "# Data distribution",
        "",
        "Run before any new Combined training. It covers class imbalance, split",
        "sizes, box scale, and the domain gap between Combined, HHU, and the small",
        "Construction val split.",
        "",
        "## How to regenerate",
        "",
        "```bash",
        "# 1) Download (requires ROBOFLOW_API_KEY; omit --execute for a dry-run)",
        "python scripts/download_datasets.py --dry-run",
        "python scripts/download_datasets.py --execute",
        "",
        "# 2) Remap onto the unified schema (never merge Construction into Combined)",
        "python scripts/remap_labels.py --source data/raw/combined \\",
        "    --out data/processed/combined --mapping combined",
        "python scripts/remap_labels.py --source data/raw/hardhat \\",
        "    --out data/processed/hardhat --mapping hhu",
        "python scripts/remap_labels.py --source data/raw/construction \\",
        "    --out data/processed/construction --mapping construction",
        "",
        "# 3) Optional 12k grid subset",
        "python scripts/make_subset.py --source data/raw/combined \\",
        "    --out data/raw/combined_12k --n 12000 --seed 42",
        "",
        "# 4) Counts, histograms, this document",
        "python scripts/analyze_distribution.py",
        "```",
        "",
        "Outputs: `results/analysis/distribution.json`, `results/analysis/class_counts_*.png`,",
        "`results/analysis/bbox_area_hist_*.png`, and this file.",
        "",
        "## Domain notes",
        "",
        "- Combined: industrial and construction mid-shot, 14 unified classes,",
        "  the only training set.",
        "- Hard Hat Universe: workplace helmet and head shots, held out for eval.",
        "  Reading `head` as `no_helmet` is an assumption.",
        "- Construction v28: small overlapping slice (val n=114), eval only.",
        "  `machinery` and `vehicle` stay here.",
        "- Construction is not merged into Combined; it already clones images from it.",
        "",
        "## Expected thin Combined classes",
        "",
        "`fall_detected`, `no_goggles`, `gloves`, and `no_gloves` are all thin,",
        "so their per-class numbers carry little weight.",
        "",
    ]

    any_present = False
    for report in reports:
        lines += [f"## {report['name']}", ""]
        if not report["present"]:
            lines += [
                f"Data not present at `{report['root']}`. Counts are pending a dataset download.",
                "",
                f"Domain: {report['domain_note']}",
                "",
            ]
            continue
        any_present = True
        lines += [
            f"Root: `{report['root']}`",
            "",
            f"Domain: {report['domain_note']}",
            "",
            "### Images per split",
            "",
            _md_table(
                ["split", "images", "empty labels"],
                [
                    [
                        split,
                        str(report["images_per_split"].get(split, 0)),
                        str(report["empty_label_images"].get(split, 0)),
                    ]
                    for split in report["images_per_split"]
                ],
            ),
            "",
            "### Boxes per class (all splits)",
            "",
        ]
        names = report["names"] or list(report["box_counts_total"])
        rows = []
        for cls in names:
            rows.append(
                [
                    cls,
                    str(report["box_counts_total"].get(cls, 0)),
                    str(report["image_presence_total"].get(cls, 0)),
                ]
            )
        lines += [_md_table(["class", "boxes", "images containing class"], rows), ""]
        lines += [
            "### Box area histogram (normalized width x height)",
            "",
            _md_table(
                ["bin", "count"],
                [
                    [f"{b['lo']:.2f}-{b['hi']:.2f}", str(b["count"])]
                    for b in report["bbox_area_hist"]
                ],
            ),
            "",
            f"n_boxes with geometry = {report['bbox_area_n']}",
            "",
            "### Imbalance flags",
            "",
        ]
        if report["imbalance_flags"]:
            lines += [f"- {flag}" for flag in report["imbalance_flags"]]
        else:
            lines.append("- None beyond the usual long tail.")
        lines += [
            "",
            f"Images with `no_*`: {report['pos_neg']['images_with_no_star']}. ",
            f"Images with positive PPE: {report['pos_neg']['images_with_positive_ppe']}.",
            "",
        ]

    if not any_present:
        lines += [
            "## Inherited Construction plot (no raw images in this checkout)",
            "",
            "The inherited `results/labels.jpg` covers the Construction training",
            "set, not Combined:",
            "",
            "- `Person` dominates the instance count, on the order of 9-10k boxes.",
            "- `machinery` and `NO-Safety Vest` are next; `Mask` and `vehicle` are the thinnest.",
            "- Most boxes are small (normalized width and height near 0). Centers",
            "  fall into a four-quadrant pattern left over from mosaic augmentation.",
            "",
            "Treat those as qualitative. Exact Combined / HHU counts require the downloads above.",
            "",
        ]

    if plot_paths:
        lines += ["## Plots", ""]
        for path in plot_paths:
            rel = Path(path)
            with contextlib.suppress(ValueError):
                rel = rel.relative_to(REPO_ROOT)
            lines.append(f"- `{rel.as_posix()}`")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    reports = []
    for name in args.datasets:
        root = DATASET_DEFAULTS.get(name, Path(name))
        if not root.is_absolute():
            root = REPO_ROOT / root
        print(f"Analyzing {name} at {root} ...")
        report = analyze_dataset(name, root)
        reports.append(report)
        print(
            f"  present={report['present']} splits={report['images_per_split']} "
            f"flags={len(report['imbalance_flags'])}"
        )

    plot_paths = _maybe_plots(reports)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    payload = [{k: v for k, v in r.items() if k != "areas"} for r in reports]
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_md(reports, plot_paths), encoding="utf-8")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
