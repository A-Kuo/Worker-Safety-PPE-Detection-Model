#!/usr/bin/env python3
"""Detection calibration: ECE, Brier, reliability, and ``no_*`` recall sweeps.

Writes ``results/analysis/calibration.json``. Operating-point rule: among
thresholds with recall >= 0.90 (or the best achievable recall), pick the
highest precision and report the precision cost.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _common import (  # noqa: E402
    NO_STAR_CLASSES,
    REPO_ROOT,
    box_iou_xyxy,
    discover_split_dirs,
    iter_split_images,
    label_path_for_image,
    load_schema,
    load_yaml,
    parse_yolo_label,
    read_dataset_names,
    yolo_norm_to_xyxy,
)
from eval import resolve_data, resolve_weights  # noqa: E402

TARGET_RECALL = 0.90


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001, help="Low dump threshold; sweep happens after.")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--max-images", type=int, default=0, help="0 = all images in the split")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "analysis" / "calibration.json")
    return parser.parse_args()


def ece_brier(confidences: list[float], correct: list[bool], n_bins: int) -> dict[str, Any]:
    if not confidences:
        return {"ece": None, "brier": None, "n": 0, "reliability": []}
    brier = sum((c - (1.0 if ok else 0.0)) ** 2 for c, ok in zip(confidences, correct)) / len(confidences)
    edges = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    reliability = []
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        idxs = [
            j
            for j, c in enumerate(confidences)
            if (c >= lo if i == 0 else c > lo) and c <= hi
        ]
        if not idxs:
            reliability.append({"lo": lo, "hi": hi, "n": 0, "confidence": None, "accuracy": None})
            continue
        acc = sum(1 for j in idxs if correct[j]) / len(idxs)
        conf = sum(confidences[j] for j in idxs) / len(idxs)
        ece += abs(acc - conf) * (len(idxs) / n)
        reliability.append({"lo": lo, "hi": hi, "n": len(idxs), "confidence": conf, "accuracy": acc})
    return {"ece": ece, "brier": brier, "n": n, "reliability": reliability}


def match_by_class(
    preds: list[tuple[int, float, tuple[float, float, float, float]]],
    gts: list[tuple[int, tuple[float, float, float, float]]],
    iou_thr: float,
) -> tuple[list[bool], list[bool]]:
    """Greedy one-to-one match. Returns (pred_is_tp, gt_is_hit) aligned to inputs."""
    pred_tp = [False] * len(preds)
    gt_hit = [False] * len(gts)
    candidates: list[tuple[float, int, int]] = []
    for i, (pc, _conf, pbox) in enumerate(preds):
        for j, (gc, gbox) in enumerate(gts):
            if pc != gc:
                continue
            iou = box_iou_xyxy(pbox, gbox)
            if iou >= iou_thr:
                candidates.append((iou, i, j))
    candidates.sort(reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    for _iou, i, j in candidates:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        pred_tp[i] = True
        gt_hit[j] = True
    return pred_tp, gt_hit


def collect_predictions(weights: Path, images: list[Path], names: list[str], imgsz: int, conf: float) -> dict[Path, list[tuple[int, float, tuple[float, float, float, float]]]]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out: dict[Path, list[tuple[int, float, tuple[float, float, float, float]]]] = {}
    model_names = None
    for image in images:
        result = model.predict(str(image), imgsz=imgsz, conf=conf, verbose=False)[0]
        if model_names is None:
            model_names = {int(k): str(v) for k, v in dict(result.names).items()}
        dets = []
        for box in result.boxes:
            cls_id = int(box.cls.item())
            # Prefer unified / data yaml name alignment by name if ids differ.
            name = (model_names or {}).get(cls_id, str(cls_id))
            if name in names:
                cls_id = names.index(name)
            dets.append((cls_id, float(box.conf.item()), tuple(float(x) for x in box.xyxy[0].tolist())))
        out[image] = dets
    return out


def load_gt(image: Path, img_w: int, img_h: int) -> list[tuple[int, tuple[float, float, float, float]]]:
    gts = []
    for row in parse_yolo_label(label_path_for_image(image)):
        if len(row) < 5:
            continue
        cls_id = int(row[0])
        x1, y1, x2, y2 = yolo_norm_to_xyxy(row[1], row[2], row[3], row[4])
        gts.append((cls_id, (x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h)))
    return gts


def _image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.size
    except Exception:  # noqa: BLE001
        import cv2

        arr = cv2.imread(str(path))
        if arr is None:
            return 640, 640
        h, w = arr.shape[:2]
        return w, h


def sweep_no_star(
    images: list[Path],
    preds: dict[Path, list],
    names: list[str],
    iou_thr: float,
    target_recall: float,
) -> dict[str, Any]:
    name_to_id = {n: i for i, n in enumerate(names)}
    thresholds = [round(x * 0.05, 2) for x in range(1, 20)]
    report: dict[str, Any] = {}
    for cls_name in NO_STAR_CLASSES:
        if cls_name not in name_to_id:
            # also accept raw Combined names
            alt = {
                "no_helmet": ("NO-Hardhat", "NO-Hardhat"),
                "no_vest": ("NO-Safety Vest",),
                "no_mask": ("NO-Mask",),
                "no_goggles": ("NO-Goggles",),
                "no_gloves": ("NO-Gloves",),
            }.get(cls_name, ())
            cls_id = next((name_to_id[a] for a in alt if a in name_to_id), None)
        else:
            cls_id = name_to_id[cls_name]
        if cls_id is None:
            report[cls_name] = {"skipped": True, "reason": "class not in dataset names"}
            continue

        rows = []
        for thr in thresholds:
            tp = fp = fn = 0
            for image in images:
                w, h = _image_size(image)
                gts = [g for g in load_gt(image, w, h) if g[0] == cls_id]
                pds = [p for p in preds.get(image, []) if p[0] == cls_id and p[1] >= thr]
                pred_tp, gt_hit = match_by_class(pds, gts, iou_thr)
                tp += sum(pred_tp)
                fp += sum(1 for ok in pred_tp if not ok)
                fn += sum(1 for ok in gt_hit if not ok)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            rows.append({"threshold": thr, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall})

        eligible = [r for r in rows if r["recall"] >= target_recall]
        if eligible:
            chosen = max(eligible, key=lambda r: (r["precision"], r["threshold"]))
            rule = f"recall>={target_recall:.2f}, max precision"
        else:
            chosen = max(rows, key=lambda r: (r["recall"], r["precision"])) if rows else None
            rule = f"best achievable recall (none reached {target_recall:.2f})"
        report[cls_name] = {
            "class_id": cls_id,
            "target_recall": target_recall,
            "rule": rule,
            "operating_point": chosen,
            "precision_cost": (1.0 - chosen["precision"]) if chosen else None,
            "curve": rows,
        }
    return report


def maybe_reliability_plots(global_cal: dict, per_class: dict, out_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    written = []
    out_dir.mkdir(parents=True, exist_ok=True)

    def _plot(rel, title, path):
        xs = [b["confidence"] for b in rel if b["n"] and b["confidence"] is not None]
        ys = [b["accuracy"] for b in rel if b["n"] and b["accuracy"] is not None]
        if not xs:
            return
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.plot(xs, ys, marker="o")
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path))

    _plot(global_cal.get("reliability") or [], "reliability (all dets)", out_dir / "reliability_global.png")
    for name, payload in per_class.items():
        _plot(payload.get("reliability") or [], f"reliability {name}", out_dir / f"reliability_{name}.png")
    return written


def main() -> int:
    args = _parse_args()
    weights = resolve_weights(args.weights)
    data = resolve_data(args.data)
    if weights is None:
        raise SystemExit("No weights found. Pass --weights after a Combined training run.")
    if data is None:
        raise SystemExit("No Combined data.yaml found. Pass --data.")

    payload_yaml = load_yaml(data)
    root = Path(payload_yaml.get("path", data.parent))
    names = read_dataset_names(data)
    splits = discover_split_dirs(root)
    images_dir = splits.get(args.split) or splits.get("valid")
    if images_dir is None:
        raise SystemExit(f"No images for split={args.split} under {root}")
    images = list(iter_split_images(images_dir))
    if args.max_images:
        images = images[: args.max_images]
    if not images:
        stub = {
            "skipped": True,
            "reason": "No images on disk for calibration. Download Combined and remap first.",
            "weights": str(weights),
            "data": str(data),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(stub, indent=2), encoding="utf-8")
        print(f"Wrote stub {args.out}")
        return 0

    print(f"Calibrating {weights} on {len(images)} images from {data}")
    preds = collect_predictions(weights, images, names, args.imgsz, args.conf)

    all_conf: list[float] = []
    all_ok: list[bool] = []
    per_cls_conf: dict[int, list[float]] = defaultdict(list)
    per_cls_ok: dict[int, list[bool]] = defaultdict(list)
    for image in images:
        w, h = _image_size(image)
        gts = load_gt(image, w, h)
        pds = preds.get(image, [])
        pred_tp, _ = match_by_class(pds, gts, args.iou)
        for (cls_id, conf, _box), ok in zip(pds, pred_tp):
            all_conf.append(conf)
            all_ok.append(ok)
            per_cls_conf[cls_id].append(conf)
            per_cls_ok[cls_id].append(ok)

    global_cal = ece_brier(all_conf, all_ok, args.bins)
    per_class = {}
    for cls_id, confs in per_cls_conf.items():
        label = names[cls_id] if cls_id < len(names) else str(cls_id)
        per_class[label] = ece_brier(confs, per_cls_ok[cls_id], args.bins)

    sweeps = sweep_no_star(images, preds, names, args.iou, TARGET_RECALL)
    plots = maybe_reliability_plots(global_cal, per_class, args.out.parent)

    try:
        schema = load_schema()
        unified = list(schema.UNIFIED_CLASS_NAMES)
    except ImportError:
        unified = names

    payload = {
        "weights": str(weights),
        "data": str(data),
        "split": args.split,
        "n_images": len(images),
        "iou": args.iou,
        "target_recall": TARGET_RECALL,
        "class_names": unified,
        "global": global_cal,
        "per_class": per_class,
        "no_star_thresholds": sweeps,
        "plots": plots,
        "note": (
            "ECE/Brier treat each detection as a binary event (IoU-matched TP vs FP). "
            "False negatives have no confidence and appear in the no_* recall sweep instead."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"ECE={global_cal['ece']} Brier={global_cal['brier']} n={global_cal['n']}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
