#!/usr/bin/env python3
"""Evaluate a unified PPE detector on Combined test (or any remapped yaml).

Prefers Ultralytics ``val`` for mAP. Optionally runs ``PPEDetector`` and
``associate_ppe_to_persons`` for a compliance summary.
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

from _common import (  # noqa: E402
    REPO_ROOT,
    first_existing,
    load_compliance,
    load_inference,
    load_schema,
    resolve_baseline_weights,
)

DEFAULT_DATA = [
    REPO_ROOT / "data" / "processed" / "combined" / "data.yaml",
    REPO_ROOT / "data" / "raw" / "combined" / "data.yaml",
    REPO_ROOT / "configs" / "data" / "combined.yaml",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "results" / "analysis" / "eval_combined.json"
    )
    parser.add_argument("--skip-compliance", action="store_true")
    parser.add_argument("--max-compliance-images", type=int, default=64)
    return parser.parse_args()


def resolve_data(override: Path | None) -> Path | None:
    if override:
        path = override if override.is_absolute() else REPO_ROOT / override
        return path if path.exists() else None
    return first_existing(*DEFAULT_DATA)


def resolve_weights(override: Path | None) -> Path | None:
    if override:
        path = override if override.is_absolute() else REPO_ROOT / override
        return path if path.exists() else None
    return first_existing(
        REPO_ROOT / "runs" / "train" / "e4_full44k" / "weights" / "best.pt",
        REPO_ROOT / "runs" / "train" / "e0_n" / "weights" / "best.pt",
        resolve_baseline_weights(),
    )


def ultralytics_val(
    weights: Path, data: Path, split: str, imgsz: int, conf: float, iou: float
) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data),
        split=split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        plots=False,
        verbose=False,
    )
    box = metrics.box
    names = {int(k): str(v) for k, v in dict(metrics.names).items()}
    per_class = []
    indices = list(getattr(box, "ap_class_index", []))
    precision = list(getattr(box, "p", []))
    recall = list(getattr(box, "r", []))
    ap50 = list(getattr(box, "ap50", []))
    ap = list(getattr(box, "ap", []))
    for i, cls_id in enumerate(indices):
        p = float(precision[i]) if i < len(precision) else float("nan")
        r = float(recall[i]) if i < len(recall) else float("nan")
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        per_class.append(
            {
                "id": int(cls_id),
                "name": names.get(int(cls_id), str(cls_id)),
                "precision": p,
                "recall": r,
                "f1": f1,
                "map50": float(ap50[i]) if i < len(ap50) else float("nan"),
                "map50_95": float(ap[i]) if i < len(ap) else float("nan"),
            }
        )
    return {
        "precision": float(getattr(box, "mp", float("nan"))),
        "recall": float(getattr(box, "mr", float("nan"))),
        "map50": float(getattr(box, "map50", float("nan"))),
        "map50_95": float(getattr(box, "map", float("nan"))),
        "per_class": per_class,
    }


def compliance_summary(
    weights: Path, data: Path, split: str, imgsz: int, conf: float, limit: int
) -> dict[str, Any]:
    from _common import discover_split_dirs, iter_split_images, load_yaml

    inference = load_inference()
    compliance = load_compliance()
    Detection = compliance.Detection
    associate = compliance.associate_ppe_to_persons
    detector = inference.PPEDetector(str(weights), conf=conf)

    yaml_dir = data.parent
    payload = load_yaml(data)
    root = Path(payload.get("path", yaml_dir))
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    splits = discover_split_dirs(root)
    images_dir = splits.get(split) or splits.get("valid") or splits.get("train")
    if images_dir is None:
        return {"skipped": True, "reason": f"no images for split={split}"}

    n_images = n_workers = n_missing = 0
    missing_hist: dict[str, int] = {}
    for i, image in enumerate(iter_split_images(images_dir)):
        if i >= limit:
            break
        n_images += 1
        try:
            dets = detector.predict_image(str(image))
        except Exception:  # noqa: BLE001
            from ultralytics import YOLO

            yres = YOLO(str(weights)).predict(str(image), imgsz=imgsz, conf=conf, verbose=False)[0]
            dets = []
            names = yres.names
            for box in yres.boxes:
                cls_id = int(box.cls.item())
                dets.append(
                    Detection(
                        cls_name=str(names.get(cls_id, cls_id)),
                        conf=float(box.conf.item()),
                        xyxy=tuple(float(x) for x in box.xyxy[0].tolist()),
                    )
                )
        workers = associate(dets)
        n_workers += len(workers)
        for worker in workers:
            missing = list(getattr(worker, "missing", []) or [])
            if missing:
                n_missing += 1
            for item in missing:
                missing_hist[str(item)] = missing_hist.get(str(item), 0) + 1

    return {
        "images": n_images,
        "workers": n_workers,
        "workers_with_missing_ppe": n_missing,
        "missing_counts": missing_hist,
    }


def main() -> int:
    args = _parse_args()
    weights = resolve_weights(args.weights)
    data = resolve_data(args.data)
    if weights is None:
        raise SystemExit("No weights found. Pass --weights path/to/best.pt")
    if data is None:
        raise SystemExit(
            "No Combined data.yaml found. Remap Combined first "
            "(data/processed/combined/data.yaml) or pass --data."
        )

    print(f"Evaluating {weights} on {data} split={args.split}")
    try:
        metrics = ultralytics_val(weights, data, args.split, args.imgsz, args.conf, args.iou)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Ultralytics val failed (missing images?): {exc}") from exc

    payload: dict[str, Any] = {
        "domain": "combined",
        "weights": str(weights),
        "data": str(data),
        "split": args.split,
        "metrics": metrics,
    }
    if not args.skip_compliance:
        try:
            payload["compliance"] = compliance_summary(
                weights, data, args.split, args.imgsz, args.conf, args.max_compliance_images
            )
        except ImportError as exc:
            payload["compliance"] = {"skipped": True, "reason": str(exc)}
        except Exception as exc:  # noqa: BLE001
            payload["compliance"] = {"skipped": True, "reason": f"{type(exc).__name__}: {exc}"}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"mAP50={metrics['map50']:.3f} mAP50-95={metrics['map50_95']:.3f} "
        f"P={metrics['precision']:.3f} R={metrics['recall']:.3f}"
    )
    print(f"Wrote {args.out}")
    try:
        schema = load_schema()
        shared = list(getattr(schema, "SHARED_EVAL_CLASSES", []))
        if shared:
            print(f"Shared eval classes (for Construction comparisons): {shared}")
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
