#!/usr/bin/env python3
"""Verify a checkpoint produced by ``reorder_checkpoint_classes.py`` is correct.

Runs both the original and reordered checkpoints on the same real image and
confirms: same number of detections, same box coordinates, same confidences
(within float tolerance) — only the class id/name differs, and the reordered
checkpoint's class name (read directly from its own ``model.names``) must
equal the original's name after mapping through ``ppe.schema``'s raw-to-unified
tables. A mismatch here means the permutation was computed or applied wrong —
do not trust the reordered checkpoint if this reports FAIL.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _common import REPO_ROOT, load_schema, mapped_unified_names  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--reordered", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--xyxy-tol", type=float, default=0.5, help="Pixel tolerance for box coords")
    parser.add_argument("--conf-tol", type=float, default=1e-4)
    return parser.parse_args()


def _predict(weights: Path, image: Path, conf: float):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    result = model.predict(str(image), conf=conf, verbose=False)[0]
    names = {int(k): str(v) for k, v in dict(result.names).items()}
    boxes = []
    for box in result.boxes:
        cls_id = int(box.cls.item())
        boxes.append(
            {
                "raw_name": names[cls_id],
                "conf": float(box.conf.item()),
                "xyxy": tuple(float(x) for x in box.xyxy[0].tolist()),
            }
        )
    # Sort by (xyxy) for stable pairwise comparison regardless of internal ordering.
    boxes.sort(key=lambda d: d["xyxy"])
    return boxes


def main() -> int:
    args = _parse_args()
    original = args.original if args.original.is_absolute() else REPO_ROOT / args.original
    reordered = args.reordered if args.reordered.is_absolute() else REPO_ROOT / args.reordered
    image = args.image if args.image.is_absolute() else REPO_ROOT / args.image

    schema = load_schema()

    orig_boxes = _predict(original, image, args.conf)
    new_boxes = _predict(reordered, image, args.conf)

    ok = True
    if len(orig_boxes) != len(new_boxes):
        print(f"FAIL: detection count differs: original={len(orig_boxes)} reordered={len(new_boxes)}")
        ok = False

    for i, (ob, nb) in enumerate(zip(orig_boxes, new_boxes, strict=False)):
        expected_unified = mapped_unified_names(schema, [ob["raw_name"]])[0]
        if nb["raw_name"] != expected_unified:
            print(
                f"FAIL[{i}]: name mismatch — original {ob['raw_name']!r} maps to "
                f"{expected_unified!r}, but reordered checkpoint says {nb['raw_name']!r}"
            )
            ok = False
        if abs(ob["conf"] - nb["conf"]) > args.conf_tol:
            print(f"FAIL[{i}]: confidence differs — {ob['conf']} vs {nb['conf']}")
            ok = False
        box_diff = max(abs(a - b) for a, b in zip(ob["xyxy"], nb["xyxy"], strict=False))
        if box_diff > args.xyxy_tol:
            print(f"FAIL[{i}]: box coords differ by {box_diff}px — {ob['xyxy']} vs {nb['xyxy']}")
            ok = False

    if ok:
        print(f"PASS: {len(orig_boxes)} detections match (boxes, confidences) with correctly relabeled classes.")
        for ob, nb in zip(orig_boxes, new_boxes, strict=False):
            print(f"  {ob['raw_name']!r} -> {nb['raw_name']!r}  conf={nb['conf']:.3f}")
        return 0
    print("VERIFICATION FAILED — do not use the reordered checkpoint.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
