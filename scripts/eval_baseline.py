#!/usr/bin/env python3
"""Audit the inherited Construction YOLOv8n baseline (Snehil Sanyal).

Loads ``baselines/snehilsanyal_yolov8n_css/models/best.pt`` (fallback
``models/best.pt``), runs Ultralytics val if Construction images exist, and
always writes per-epoch global metrics from ``results.csv`` into
``docs/baseline.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _common import (  # noqa: E402
    CONSTRUCTION_CLASS_NAMES,
    KNOWN_EPOCH99,
    REPO_ROOT,
    count_images,
    find_construction_image_root,
    find_construction_yaml,
    fmt,
    parse_results_csv,
    resolve_baseline_weights,
    resolve_confusion_matrix,
    resolve_results_csv,
)

DOCS_BASELINE = REPO_ROOT / "docs" / "baseline.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None, help="Construction data.yaml")
    parser.add_argument("--results-csv", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DOCS_BASELINE)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", default="val")
    parser.add_argument("--skip-val", action="store_true")
    return parser.parse_args()


def _metric(row: dict[str, float], *keys: str) -> float | None:
    for key in keys:
        if key in row:
            return row[key]
    return None


def run_ultralytics_val(weights: Path, data_yaml: Path, imgsz: int, split: str) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), split=split, imgsz=imgsz, plots=False, verbose=False)
    box = metrics.box
    names = {int(k): str(v) for k, v in dict(metrics.names).items()}
    per_class = []
    indices = list(getattr(box, "ap_class_index", range(len(names))))
    precision = list(getattr(box, "p", []))
    recall = list(getattr(box, "r", []))
    ap50 = list(getattr(box, "ap50", []))
    ap = list(getattr(box, "ap", []))
    for i, cls_id in enumerate(indices):
        p = float(precision[i]) if i < len(precision) else float("nan")
        r = float(recall[i]) if i < len(recall) else float("nan")
        a50 = float(ap50[i]) if i < len(ap50) else float("nan")
        a = float(ap[i]) if i < len(ap) else float("nan")
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        per_class.append(
            {
                "id": int(cls_id),
                "name": names.get(int(cls_id), str(cls_id)),
                "precision": p,
                "recall": r,
                "f1": f1,
                "map50": a50,
                "map50_95": a,
            }
        )
    return {
        "precision": float(getattr(box, "mp", float("nan"))),
        "recall": float(getattr(box, "mr", float("nan"))),
        "map50": float(getattr(box, "map50", float("nan"))),
        "map50_95": float(getattr(box, "map", float("nan"))),
        "per_class": per_class,
    }


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_baseline_md(
    *,
    weights: Path | None,
    results_csv: Path | None,
    epochs: list[dict[str, float]],
    val_metrics: dict | None,
    val_error: str | None,
    image_root: Path | None,
    data_yaml: Path | None,
    n_val_images: int,
) -> str:
    last = epochs[-1] if epochs else {}
    last_epoch = int(last.get("epoch", 99))
    p = _metric(last, "metrics/precision(B)")
    r = _metric(last, "metrics/recall(B)")
    m50 = _metric(last, "metrics/mAP50(B)")
    m95 = _metric(last, "metrics/mAP50-95(B)")

    best_m50 = (
        max(epochs, key=lambda row: _metric(row, "metrics/mAP50(B)") or -1) if epochs else None
    )
    best_m95 = (
        max(epochs, key=lambda row: _metric(row, "metrics/mAP50-95(B)") or -1) if epochs else None
    )

    lines = [
        "# Construction YOLOv8n baseline",
        "",
        "A third-party baseline, not a model trained in this repo.",
        "Weights, Ultralytics plots, and `results.csv` were inherited from",
        "[snehilsanyal/Construction-Site-Safety-PPE-Detection](https://github.com/snehilsanyal/Construction-Site-Safety-PPE-Detection)",
        "by Snehil Sanyal. Credit Snehil Sanyal and the Roboflow Construction Site",
        "Safety v28 dataset (CC BY 4.0).",
        "",
        "This 10-class Construction mAP is not comparable to a 14-class Combined",
        "model. The vocabularies differ, so the numbers measure different tasks.",
        "",
        "## Headline numbers (epoch 99)",
        "",
        "Cited values from the inherited training log (rounded):",
        "",
        f"- **mAP@0.50** = {KNOWN_EPOCH99['map50']:.3f}",
        f"- **mAP@0.50:0.95** = {KNOWN_EPOCH99['map50_95']:.3f}",
        f"- **Precision** = {KNOWN_EPOCH99['precision']:.3f}",
        f"- **Recall** = {KNOWN_EPOCH99['recall']:.3f}",
        "",
    ]
    if last:
        lines += [
            "Exact `results.csv` values at the final logged epoch "
            f"({last_epoch}): "
            f"P={fmt(p or 0, 5)}, R={fmt(r or 0, 5)}, "
            f"mAP50={fmt(m50 or 0, 5)}, mAP50-95={fmt(m95 or 0, 5)}.",
            "",
        ]
    if best_m50 and best_m95:
        lines += [
            f"Best logged mAP50 is epoch {int(best_m50.get('epoch', -1))} "
            f"({fmt(_metric(best_m50, 'metrics/mAP50(B)') or 0, 5)}). "
            f"Best mAP50-95 is epoch {int(best_m95.get('epoch', -1))} "
            f"({fmt(_metric(best_m95, 'metrics/mAP50-95(B)') or 0, 5)}).",
            "",
        ]

    lines += [
        "## Dataset and split caveat",
        "",
        "- Dataset: Roboflow Universe Construction Site Safety v28,",
        "  https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/28",
        "- Classes (10): " + ", ".join(f"`{n}`" for n in CONSTRUCTION_CLASS_NAMES) + ".",
        "- Published split: 2605 / 114 / 82 (train / val / test).",
        "- Per-class precision, recall, F1, and mAP over 114 val images are noisy.",
        "  A single-class swing there is not a modeling result. Test (n=82) is smaller still.",
        "- `machinery` and `vehicle` exist only on this baseline. Neither is in the",
        "  unified 14-class schema.",
        "",
        "## Artifacts used",
        "",
        f"- Weights: `{weights.as_posix() if weights else 'not found'}`",
        f"- Training log: `{results_csv.as_posix() if results_csv else 'not found'}`",
        f"- Construction yaml: `{data_yaml.as_posix() if data_yaml else 'not found'}`",
        f"- Image root: `{image_root.as_posix() if image_root else 'not found'}`"
        f" ({n_val_images} images discovered)",
        "",
        "## Ultralytics val (per-class)",
        "",
    ]

    if val_metrics:
        lines += [
            "Fresh `model.val()` on the Construction split:",
            "",
            f"- Precision = {fmt(val_metrics['precision'])}",
            f"- Recall = {fmt(val_metrics['recall'])}",
            f"- mAP50 = {fmt(val_metrics['map50'])}",
            f"- mAP50-95 = {fmt(val_metrics['map50_95'])}",
            "",
            _md_table(
                ["class", "P", "R", "F1", "mAP50", "mAP50-95"],
                [
                    [
                        row["name"],
                        fmt(row["precision"]),
                        fmt(row["recall"]),
                        fmt(row["f1"]),
                        fmt(row["map50"]),
                        fmt(row["map50_95"]),
                    ]
                    for row in val_metrics["per_class"]
                ],
            ),
            "",
        ]
    else:
        lines += [
            "Per-class P / R / F1 / mAP need a val pass with Construction images on",
            "disk, and this checkout has none (`data/raw/construction` and the",
            "original train/valid/test folders are empty or missing).",
            "",
            "The inherited `results.csv` stores global box metrics only, so per-class",
            "numbers cannot be recovered from it.",
            "Re-run:",
            "",
            "```bash",
            "python scripts/eval_baseline.py",
            "```",
            "",
            "after placing Construction v28 images (or setting `--data` / `--weights`).",
            "",
        ]
        if val_error:
            lines += [f"Val attempt error: `{val_error}`", ""]

    cm = resolve_confusion_matrix()
    lines += [
        "## Confusion-matrix interpretation (inherited plot)",
        "",
        "Read from the shipped "
        f"`{cm.as_posix() if cm else 'results/confusion_matrix.png'}` "
        "(normalized, and no substitute for a real per-class val pass):",
        "",
        "- Strongest diagonal (recall-like): `machinery` ~0.93, `Safety Cone` ~0.91, `Mask` ~0.90.",
        "- Mid: `Person` ~0.80, `Safety Vest` ~0.78, `Hardhat` ~0.76.",
        "- Weakest: `NO-Safety Vest` ~0.70, `NO-Mask` ~0.66, `NO-Hardhat` ~0.62, `vehicle` ~0.57.",
        "- The `NO-*` classes leak into background, which means missed violations:",
        "  `NO-Hardhat` ~0.36, `NO-Mask` ~0.34.",
        "- `vehicle` is often missed entirely (~0.43 background FN), and `Person`",
        "  has a high background false-positive rate (~0.29).",
        "- Some `NO-Safety Vest` boxes come back as `Safety Vest` (~0.07). That",
        "  polarity flip is the most expensive error a compliance system can make.",
        "",
        "Hence the confidence sweeps on `no_*` and the recall-first operating points",
        "in the calibration step.",
        "",
        "## Per-epoch global metrics",
        "",
        "Parsed from the inherited Ultralytics `results.csv` (one row per epoch). "
        "These are **global** box metrics, not per-class.",
        "",
    ]

    if epochs:
        table_rows = []
        for row in epochs:
            table_rows.append(
                [
                    str(int(row.get("epoch", -1))),
                    fmt(_metric(row, "metrics/precision(B)") or 0, 4),
                    fmt(_metric(row, "metrics/recall(B)") or 0, 4),
                    fmt(_metric(row, "metrics/mAP50(B)") or 0, 4),
                    fmt(_metric(row, "metrics/mAP50-95(B)") or 0, 4),
                    fmt(_metric(row, "val/box_loss") or 0, 4),
                    fmt(_metric(row, "val/cls_loss") or 0, 4),
                    fmt(_metric(row, "val/dfl_loss") or 0, 4),
                ]
            )
        lines.append(
            _md_table(
                ["epoch", "P", "R", "mAP50", "mAP50-95", "val/box", "val/cls", "val/dfl"],
                table_rows,
            )
        )
        lines.append("")
    else:
        lines += ["`results.csv` was not found; cannot list per-epoch metrics.", ""]

    lines += [
        "## How this was generated",
        "",
        "```bash",
        "python scripts/eval_baseline.py",
        "```",
        "",
        "The script rewrites this file from `results.csv`. Per-class rows appear",
        "only when val images are present.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    weights = args.weights or resolve_baseline_weights()
    results_csv = args.results_csv or resolve_results_csv()
    data_yaml = args.data or find_construction_yaml()
    image_root = find_construction_image_root()
    n_images = count_images(image_root) if image_root else 0

    epochs: list[dict[str, float]] = []
    if results_csv and results_csv.exists():
        epochs = parse_results_csv(results_csv)
    else:
        print(
            "WARNING: results.csv not found; writing known epoch-99 numbers only.", file=sys.stderr
        )

    val_metrics = None
    val_error = None
    can_val = (
        not args.skip_val
        and weights is not None
        and weights.exists()
        and data_yaml is not None
        and data_yaml.exists()
        and n_images > 0
    )
    if can_val:
        try:
            print(f"Running Ultralytics val: weights={weights} data={data_yaml} images={n_images}")
            val_metrics = run_ultralytics_val(weights, data_yaml, args.imgsz, args.split)
        except Exception as exc:  # noqa: BLE001 - the val pass is best-effort
            val_error = f"{type(exc).__name__}: {exc}"
            print(f"Val failed: {val_error}", file=sys.stderr)
    else:
        print("Skipping val (no Construction images, missing weights/yaml, or --skip-val).")

    text = render_baseline_md(
        weights=weights,
        results_csv=results_csv,
        epochs=epochs,
        val_metrics=val_metrics,
        val_error=val_error,
        image_root=image_root,
        data_yaml=data_yaml,
        n_val_images=n_images,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"Wrote {args.out}")
    if epochs:
        last = epochs[-1]
        print(
            "epoch {e}: P={p:.3f} R={r:.3f} mAP50={m50:.3f} mAP50-95={m95:.3f}".format(
                e=int(last.get("epoch", -1)),
                p=_metric(last, "metrics/precision(B)") or 0,
                r=_metric(last, "metrics/recall(B)") or 0,
                m50=_metric(last, "metrics/mAP50(B)") or 0,
                m95=_metric(last, "metrics/mAP50-95(B)") or 0,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
