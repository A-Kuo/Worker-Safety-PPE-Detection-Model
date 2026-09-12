#!/usr/bin/env python3
"""Permute a YOLO checkpoint's detection-head class order to match a target order.

Root problem this fixes: Ultralytics matches predictions to ground truth by
**numeric class id**, not name. An externally-sourced checkpoint (e.g.
Hexmon/vyra-yolo-ppe-detection) may order its classes differently from this
repo's ``UNIFIED_CLASS_NAMES`` (e.g. alphabetically) even when it covers the
exact same 14 concepts. That silently corrupts both:

- ``scripts/eval.py``'s raw ``model.val()`` mAP (confirmed empirically —
  see ``docs/baseline_hf.md``: Hexmon scored mAP50=0.057 raw, but a
  name-based spot-check showed it correctly detecting helmets at 0.87 conf).
- ``scripts/finetune.py`` fine-tuning against a standard unified-order
  dataset yaml (its schema guard correctly refuses this — see
  ``check_schema_compatibility``).

This script permutes ONLY the final classification conv layer's output
channels (weight + bias, dim 0) in each of YOLOv8's three detection-scale
``cv3`` branches — the box-regression ``cv2`` branches are untouched since
they don't depend on class order. All other weights are bit-identical;
this is a pure relabeling, not a retrain. Verify with
``scripts/verify_reorder.py`` (compares detections before/after on a real
image — same boxes/confidences, only class ids/names change).
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
    parser.add_argument("--weights", type=Path, required=True, help="Source checkpoint (.pt)")
    parser.add_argument("--out", type=Path, required=True, help="Destination checkpoint (.pt)")
    parser.add_argument(
        "--target-order",
        nargs="+",
        default=None,
        help="Target class name order (default: schema.UNIFIED_CLASS_NAMES)",
    )
    return parser.parse_args()


def _compute_permutation(raw_names: list[str], mapped_names: list[str], target_order: list[str]) -> list[int]:
    """perm[i] = source index whose mapped concept equals target_order[i]."""
    if len(mapped_names) != len(target_order):
        raise SystemExit(
            f"Class count mismatch: checkpoint has {len(mapped_names)} classes "
            f"{mapped_names}, target order has {len(target_order)} {target_order}."
        )
    if sorted(mapped_names) != sorted(target_order):
        raise SystemExit(
            f"Class sets differ, cannot permute:\n  checkpoint (mapped): {sorted(mapped_names)}\n"
            f"  target order:        {sorted(target_order)}"
        )
    index_by_concept = {name: i for i, name in enumerate(mapped_names)}
    perm = [index_by_concept[name] for name in target_order]
    if sorted(perm) != list(range(len(perm))):
        raise SystemExit(f"Internal error: not a valid permutation: {perm}")
    return perm


def _permute_detect_head(detection_model, perm: list[int], target_names: dict[int, str]) -> None:
    """Permute cv3 (classification) output channels in-place; leave cv2 (box) untouched."""
    import torch

    head = detection_model.model[-1]
    if not hasattr(head, "cv3"):
        raise SystemExit(f"Unexpected head type (no cv3): {type(head)}")
    perm_t = torch.tensor(perm, dtype=torch.long)
    for branch in head.cv3:
        conv = branch[-1]  # final 1x1 Conv2d producing nc output channels
        if conv.weight.shape[0] != len(perm):
            raise SystemExit(
                f"Head channel count {conv.weight.shape[0]} != permutation length {len(perm)}"
            )
        conv.weight.data = conv.weight.data[perm_t].clone()
        if conv.bias is not None:
            conv.bias.data = conv.bias.data[perm_t].clone()
    detection_model.names = dict(target_names)


def main() -> int:
    args = _parse_args()
    weights = args.weights if args.weights.is_absolute() else REPO_ROOT / args.weights
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if not weights.is_file():
        raise SystemExit(f"Weights not found: {weights}")

    schema = load_schema()
    target_order = args.target_order or list(schema.UNIFIED_CLASS_NAMES)

    from ultralytics import YOLO

    model = YOLO(str(weights))
    raw_names_dict = model.names
    raw_names = [str(raw_names_dict[i]) for i in sorted(raw_names_dict)]
    mapped_names = mapped_unified_names(schema, raw_names)

    perm = _compute_permutation(raw_names, mapped_names, target_order)
    print(f"Source order (raw):    {raw_names}")
    print(f"Source order (mapped): {mapped_names}")
    print(f"Target order:           {target_order}")
    print(f"Permutation (target[i] <- source[perm[i]]): {perm}")

    target_names = {i: name for i, name in enumerate(target_order)}
    _permute_detect_head(model.model, perm, target_names)

    import torch

    raw_ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
    if raw_ckpt.get("ema") is not None:
        print("WARNING: checkpoint also has a non-None 'ema' entry; permuting it too.")
        ema_names_dict = getattr(raw_ckpt["ema"], "names", None)
        if ema_names_dict is not None:
            ema_names = [str(ema_names_dict[i]) for i in sorted(ema_names_dict)]
            if ema_names != raw_names:
                raise SystemExit(
                    "ema model's class names differ from model's — refusing to guess "
                    "a permutation for it. Inspect manually."
                )
        _permute_detect_head(raw_ckpt["ema"], perm, target_names)
        model.ckpt = raw_ckpt  # ensure save() picks up the permuted ema too

    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print(f"Wrote {out}")
    print(
        "Verify with scripts/verify_reorder.py before trusting this checkpoint for "
        "fine-tuning or evaluation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
