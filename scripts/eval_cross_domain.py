#!/usr/bin/env python3
"""Cross-domain eval: Combined test, Construction shared classes, HHU helmet/vest.

Documents the HHU ``head`` -> ``no_helmet`` assumption. Reports only shared
classes on Construction so 10-class and 14-class mAP are not compared raw.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _common import REPO_ROOT, first_existing, load_schema  # noqa: E402
from eval import resolve_weights, ultralytics_val  # noqa: E402

HHU_ASSUMPTION = (
    "Hard Hat Universe remap treats `head` as `no_helmet` (implicit missing helmet). "
    "That is an assumption, not a labeled NO-Hardhat class. Score HHU `no_helmet` "
    "separately so it does not pollute Combined metrics."
)

DOMAIN_DEFAULTS = {
    "combined": [
        REPO_ROOT / "data" / "processed" / "combined" / "data.yaml",
        REPO_ROOT / "data" / "raw" / "combined" / "data.yaml",
        REPO_ROOT / "configs" / "data" / "combined.yaml",
    ],
    "construction": [
        REPO_ROOT / "data" / "processed" / "construction" / "data.yaml",
        REPO_ROOT / "data" / "raw" / "construction" / "data.yaml",
        REPO_ROOT / "configs" / "data" / "construction.yaml",
    ],
    "hhu": [
        REPO_ROOT / "data" / "processed" / "hardhat" / "data.yaml",
        REPO_ROOT / "data" / "raw" / "hardhat" / "data.yaml",
        REPO_ROOT / "configs" / "data" / "hardhat_eval.yaml",
    ],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--combined-data", type=Path, default=None)
    parser.add_argument("--construction-data", type=Path, default=None)
    parser.add_argument("--hhu-data", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--construction-split", default="test")
    parser.add_argument("--hhu-split", default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "results" / "analysis" / "cross_domain.json"
    )
    return parser.parse_args()


def _resolve_domain(name: str, override: Path | None) -> Path | None:
    if override:
        path = override if override.is_absolute() else REPO_ROOT / override
        return path if path.exists() else None
    return first_existing(*DOMAIN_DEFAULTS[name])


def _filter_shared(per_class: list[dict], shared: list[str]) -> list[dict]:
    if not shared:
        return per_class
    allow = {n.casefold() for n in shared}
    return [row for row in per_class if row["name"].casefold() in allow]


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [row[key] for row in rows if row.get(key) == row.get(key)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def eval_domain(
    name: str,
    weights: Path,
    data: Path | None,
    split: str,
    imgsz: int,
    conf: float,
    iou: float,
    shared: list[str],
) -> dict[str, Any]:
    if data is None:
        return {
            "domain": name,
            "skipped": True,
            "reason": f"No remapped/raw yaml for {name}. Run download + remap first.",
        }
    try:
        metrics = ultralytics_val(weights, data, split, imgsz, conf, iou)
    except Exception as exc:  # noqa: BLE001
        return {
            "domain": name,
            "data": str(data),
            "skipped": True,
            "reason": f"{type(exc).__name__}: {exc}",
        }

    per_class = metrics["per_class"]
    if name == "construction":
        per_class = _filter_shared(per_class, shared)
    if name == "hhu":
        allow = {"helmet", "vest", "person", "no_helmet"}
        per_class = [row for row in per_class if row["name"].casefold() in allow]

    return {
        "domain": name,
        "data": str(data),
        "split": split,
        "skipped": False,
        "precision": metrics["precision"] if name == "combined" else _mean(per_class, "precision"),
        "recall": metrics["recall"] if name == "combined" else _mean(per_class, "recall"),
        "map50": metrics["map50"] if name == "combined" else _mean(per_class, "map50"),
        "map50_95": metrics["map50_95"] if name == "combined" else _mean(per_class, "map50_95"),
        "per_class": per_class,
        "note": HHU_ASSUMPTION if name == "hhu" else None,
    }


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    weights = resolve_weights(args.weights)
    if weights is None:
        raise SystemExit("No weights found. Pass --weights after a Combined training run.")

    try:
        schema = load_schema()
        shared = list(getattr(schema, "SHARED_EVAL_CLASSES", []))
    except ImportError as exc:
        print(f"WARNING: {exc}", file=sys.stderr)
        shared = ["helmet", "no_helmet", "vest", "no_vest", "mask", "no_mask", "person", "cone"]

    domains = {
        "combined": (_resolve_domain("combined", args.combined_data), args.split),
        "construction": (
            _resolve_domain("construction", args.construction_data),
            args.construction_split,
        ),
        "hhu": (_resolve_domain("hhu", args.hhu_data), args.hhu_split),
    }

    results = []
    for name, (data, split) in domains.items():
        print(f"== {name} ==")
        row = eval_domain(name, weights, data, split, args.imgsz, args.conf, args.iou, shared)
        results.append(row)
        if row.get("skipped"):
            print(f"  skipped: {row.get('reason')}")
        else:
            print(
                f"  mAP50={row.get('map50')} R={row.get('recall')} "
                f"classes={len(row.get('per_class') or [])}"
            )

    payload = {
        "weights": str(weights),
        "shared_eval_classes": shared,
        "hhu_assumption": HHU_ASSUMPTION,
        "domains": results,
        "tradeoff_note": (
            "One general Combined model vs a helmet-specialist. "
            "This evaluates the general model only. A domain specialist is out of",
            "scope unless E4 finishes early.",
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_path = args.out.with_suffix(".md")
    table_rows = []
    for row in results:
        if row.get("skipped"):
            table_rows.append(
                [
                    row["domain"],
                    "pending eval",
                    "pending eval",
                    "pending eval",
                    "pending eval",
                    row.get("reason", ""),
                ]
            )
        else:
            table_rows.append(
                [
                    row["domain"],
                    f"{row['map50']:.3f}" if row.get("map50") is not None else "n/a",
                    f"{row['map50_95']:.3f}" if row.get("map50_95") is not None else "n/a",
                    f"{row['precision']:.3f}" if row.get("precision") is not None else "n/a",
                    f"{row['recall']:.3f}" if row.get("recall") is not None else "n/a",
                    row.get("note") or "",
                ]
            )
    md = "\n".join(
        [
            "# Cross-domain evaluation",
            "",
            HHU_ASSUMPTION,
            "",
            "Construction numbers cover `SHARED_EVAL_CLASSES` only; raw 10-class and",
            "14-class mAP are not comparable.",
            "",
            _md_table(["domain", "mAP50", "mAP50-95", "P", "R", "notes"], table_rows),
            "",
            f"Weights: `{weights}`",
            "",
        ]
    )
    md_path.write_text(md, encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
