#!/usr/bin/env python3
"""Failure-mode breakdown for one class (default ``no_vest``) on a labeled split.

mAP says *that* a class is weak; this says *why*. For every ground-truth box of the target
class it finds what the model did there at a very low confidence floor and buckets it:

  hit>=0.25 / hit 0.05-0.25 / hit 0.01-0.05   correct class, IoU>=0.5, at that confidence band
  flipped                                     opposite class (e.g. vest for no_vest) instead
  other class                                 some other class overlaps it
  mislocalized                                right class but IoU only 0.1-0.5
  missed                                      nothing usable

It also sweeps a confidence threshold: recall on the target boxes vs how many false target-class
boxes fire per 100 images that contain none. Bucket sizes tell you the fix: many low-confidence
hits -> a threshold/calibration fix; mostly "missed" -> the model can't see it (data/capacity).

Checkpoints with raw third-party class names (e.g. Hexmon's) are mapped to unified names first.
Ground truth is read from the remapped labels (data/processed/combined), never data/raw.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _common import (  # noqa: E402
    REPO_ROOT,
    box_iou_xyxy,
    iter_split_images,
    label_path_for_image,
    load_schema,
    mapped_unified_names,
    parse_yolo_label,
    yolo_norm_to_xyxy,
)

OPPOSITE = {"no_vest": "vest", "vest": "no_vest", "no_helmet": "helmet", "helmet": "no_helmet",
            "no_goggles": "goggles", "goggles": "no_goggles", "no_gloves": "gloves",
            "gloves": "no_gloves", "no_mask": "mask", "mask": "no_mask"}
THRESHOLDS = (0.01, 0.05, 0.10, 0.25, 0.40)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, nargs="+", required=True)
    parser.add_argument("--cls", default="no_vest", help="unified class name to diagnose")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "processed" / "combined")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--negatives", type=int, default=300, help="images WITHOUT the class, for false-positive rates")
    parser.add_argument("--limit", type=int, default=0, help="cap positive images (0 = all)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def gt_boxes(image: Path, unified_index: int):
    rows = parse_yolo_label(label_path_for_image(image))
    return [(int(r[0]), yolo_norm_to_xyxy(*r[1:5])) for r in rows if len(r) >= 5]


def geometry(boxes):
    if not boxes:
        return {}
    areas = sorted((b[2] - b[0]) * (b[3] - b[1]) for b in boxes)
    aspects = sorted((b[3] - b[1]) / max(b[2] - b[0], 1e-9) for b in boxes)
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]  # noqa: E731
    return {"n": len(boxes), "area_p10": q(areas, .1), "area_median": q(areas, .5), "area_p90": q(areas, .9),
            "aspect_h_over_w_median": q(aspects, .5)}


def predict_all(model, images, imgsz, device, names_unified):
    """image path -> list[(unified_name, conf, xyxyn)] at conf>=0.01."""
    out = {}
    for start in range(0, len(images), 16):
        chunk = images[start:start + 16]
        for path, res in zip(chunk, model.predict([str(p) for p in chunk], imgsz=imgsz, conf=0.01,
                                                  device=device, verbose=False), strict=True):
            preds = []
            for box in res.boxes:
                preds.append((names_unified[int(box.cls.item())], float(box.conf.item()),
                              tuple(float(x) for x in box.xyxyn[0].tolist())))
            out[path] = preds
    return out


def bucket(gt_box, preds, target, opposite):
    best = {}
    for name, conf, box in preds:
        iou = box_iou_xyxy(gt_box, box)
        key = (name, iou >= 0.5)
        if key not in best or conf > best[key]:
            best[key] = conf
    on_target = best.get((target, True))
    if on_target is not None:
        if on_target >= 0.25:
            return "hit>=0.25"
        if on_target >= 0.05:
            return "hit 0.05-0.25"
        return "hit 0.01-0.05"
    if opposite and best.get((opposite, True), 0) >= 0.25:
        return "flipped"
    if any(ok and conf >= 0.25 for (n, ok), conf in best.items()):
        return "other class"
    if any(box_iou_xyxy(gt_box, b) >= 0.1 for n, c, b in preds if n == target):
        return "mislocalized"
    return "missed"


def main() -> int:
    args = _parse_args()
    schema = load_schema()
    unified = list(schema.UNIFIED_CLASS_NAMES)
    if args.cls not in unified:
        raise SystemExit(f"unknown class {args.cls!r}; choose from {unified}")
    target_idx = unified.index(args.cls)
    opposite = OPPOSITE.get(args.cls)

    images = list(iter_split_images(args.data_root / args.split / "images"))
    positives, negatives = [], []
    target_gt, opposite_gt = [], []
    for image in images:
        boxes = gt_boxes(image, target_idx)
        mine = [b for c, b in boxes if c == target_idx]
        if mine:
            positives.append((image, mine))
            target_gt += mine
        else:
            negatives.append(image)
        if opposite:
            opposite_gt += [b for c, b in boxes if c == unified.index(opposite)]
    if args.limit:
        positives = positives[:args.limit]
    negatives = random.Random(0).sample(negatives, min(args.negatives, len(negatives)))
    print(f"{args.cls} on {args.split}: {len(target_gt)} GT boxes in {len(positives)} images "
          f"(of {len(images)}); {len(negatives)} negative images sampled for false-positive rates")
    print("GT geometry (fractions of image): ", json.dumps({args.cls: geometry(target_gt),
                                                            **({opposite: geometry(opposite_gt)} if opposite else {})}))

    from ultralytics import YOLO

    report = {}
    for weights in args.weights:
        w = weights if weights.is_absolute() else REPO_ROOT / weights
        model = YOLO(str(w))
        raw = [str(model.names[i]) for i in sorted(model.names)]
        mapped = dict(zip(sorted(model.names), mapped_unified_names(schema, raw), strict=True))
        pos_imgs = [p for p, _ in positives]
        preds_pos = predict_all(model, pos_imgs, args.imgsz, args.device, mapped)
        preds_neg = predict_all(model, negatives, args.imgsz, args.device, mapped)

        counts = Counter()
        for image, boxes in positives:
            for gb in boxes:
                counts[bucket(gb, preds_pos[image], args.cls, opposite)] += 1
        total = sum(counts.values())
        sweep = []
        for t in THRESHOLDS:
            tp = 0
            for image, boxes in positives:
                tgt = [(c, b) for n, c, b in preds_pos[image] if n == args.cls and c >= t]
                tp += sum(1 for gb in boxes if any(box_iou_xyxy(gb, b) >= 0.5 for _, b in tgt))
            fp_neg = sum(1 for p in negatives for n, c, _ in preds_neg[p] if n == args.cls and c >= t)
            sweep.append({"conf": t, "recall": tp / max(1, total),
                          "false_boxes_per_100_negative_images": 100 * fp_neg / max(1, len(negatives))})
        report[str(weights)] = {"buckets": dict(counts), "sweep": sweep}

        print(f"\n=== {weights} ===")
        for name in ("hit>=0.25", "hit 0.05-0.25", "hit 0.01-0.05", "flipped", "other class", "mislocalized", "missed"):
            print(f"  {name:15} {counts[name]:5}  {100 * counts[name] / max(1, total):5.1f}%")
        print(f"  {'conf>=':>8} {'recall':>8} {'false ' + args.cls + ' boxes / 100 clean imgs':>34}")
        for row in sweep:
            print(f"  {row['conf']:>8.2f} {row['recall']:>8.3f} {row['false_boxes_per_100_negative_images']:>34.1f}")

    out = args.out or REPO_ROOT / "results" / "analysis" / f"diagnose_{args.cls}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
