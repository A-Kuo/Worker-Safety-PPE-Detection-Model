#!/usr/bin/env python3
"""Post-training sanity check: does an independent test eval agree with the run's own validation?

The fingerprint of a train/eval label mismatch is *disagreement*: the training-time
validation reads the same labels training used, so it looks healthy, while an
independent eval on ``data/processed/combined`` (the real remapped labels) collapses.
An absolute mAP floor can't tell that from a short or under-trained run, so this
compares the two. Call ``run("e0_n")`` from a notebook cell (raises AssertionError
on a mismatch, which aborts "Run All" before more credits are spent) or run the
script directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _common import REPO_ROOT  # noqa: E402

MAP50_KEY = "metrics/mAP50(B)"
# Score with the same protocol Ultralytics' training-time validation uses (conf 0.001, NMS IoU 0.7).
# eval.py's own defaults (conf 0.25, NMS IoU 0.5) truncate the PR curve, so a weakly-trained model
# whose scores are all below 0.25 reads 0.000 there while its training-time val reads e.g. 0.4 -
# the two numbers compared below would then disagree for a reason that has nothing to do with labels
# (found in the Linux rehearsal: a healthy 1-epoch smoke run failed this check).
EVAL_CONF = 0.001
EVAL_NMS_IOU = 0.7
# A mismatch only looks like a mismatch if the training-time val looked healthy.
MIN_HEALTHY_VAL = 0.1
AGREEMENT_RATIO = 0.5
LOW_ABSOLUTE = 0.3


def read_best_val_map50(results_csv: Path) -> tuple[float, int]:
    """Best training-time val mAP50 and the number of epochs logged."""
    with results_csv.open(newline="", encoding="utf-8") as handle:
        rows = [{(k or "").strip(): v for k, v in row.items()} for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"{results_csv} has no epoch rows")
    if MAP50_KEY not in rows[0]:
        raise ValueError(f"{results_csv} has no {MAP50_KEY!r} column: {sorted(rows[0])}")
    return max(float(r[MAP50_KEY]) for r in rows), len(rows)


def verdict(best_val: float, test: float) -> tuple[str, str]:
    """Return ("pass" | "warn" | "fail", message)."""
    if best_val >= MIN_HEALTHY_VAL and test < AGREEMENT_RATIO * best_val:
        return "fail", (
            f"test mAP50 {test:.3f} is far below the training-time val mAP50 {best_val:.3f}: the model was "
            "trained on different labels than it is being scored against (check the train list / "
            "configs/data/*.yaml resolve to data/processed/..., not data/raw/...)."
        )
    if test < LOW_ABSOLUTE:
        return "warn", (
            f"test mAP50 {test:.3f} is low, but it agrees with training-time val ({best_val:.3f}), so this "
            "looks under-trained (too few epochs?) rather than mislabeled."
        )
    return "pass", f"test mAP50 {test:.3f} and training-time val {best_val:.3f} agree; labels are consistent."


def eval_command(weights: Path, split: str, out_json: Path, data: str | None = None) -> list[str]:
    cmd = [sys.executable, str(_SCRIPTS / "eval.py"), "--weights", str(weights), "--split", split,
           "--out", str(out_json), "--conf", str(EVAL_CONF), "--iou", str(EVAL_NMS_IOU)]
    if data:
        cmd += ["--data", data]
    return cmd


def run(exp: str, project: Path | None = None, split: str = "test", data: str | None = None) -> str:
    project = Path(project) if project else REPO_ROOT / "runs" / "train"
    run_dir = project / exp
    weights = run_dir / "weights" / "best.pt"
    results_csv = run_dir / "results.csv"
    for required in (weights, results_csv):
        if not required.is_file():
            raise AssertionError(f"{required} is missing — training did not produce a checkpoint (see its log).")

    out_json = REPO_ROOT / "results" / "analysis" / f"eval_{exp}.json"
    log_path = run_dir / "sanity_eval.log"
    cmd = eval_command(weights, split, out_json, data)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:])
    print(tail)
    if proc.returncode != 0 or not out_json.is_file():
        raise AssertionError(f"eval.py failed (exit {proc.returncode}); full log: {log_path}")

    metrics = json.loads(out_json.read_text(encoding="utf-8"))["metrics"]
    best_val, epochs = read_best_val_map50(results_csv)
    test = float(metrics["map50"])
    print(f"\nepochs trained: {epochs}")
    print(f"training-time val mAP50 (best epoch): {best_val:.3f}")
    print(f"independent {split} mAP50: {test:.3f}   mAP50-95={metrics['map50_95']:.3f} "
          f"P={metrics['precision']:.3f} R={metrics['recall']:.3f}")

    status, message = verdict(best_val, test)
    if status == "fail":
        for name in sorted((REPO_ROOT / "data" / "raw").glob("combined_*/train.txt")):
            head = name.read_text(encoding="utf-8").splitlines()[:2]
            print(f"\nDIAGNOSTIC {name} (entries must live under data/processed/, never data/raw/combined/):")
            print("\n".join(head))
        raise AssertionError(message + " Do NOT copy this checkpoint off the VM — investigate first.")
    print(("\nWARNING: " if status == "warn" else "\nSanity check passed - ") + message)
    return status


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--project", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--data", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        run(args.exp, args.project, args.split, args.data)
    except AssertionError as exc:
        raise SystemExit(f"SANITY CHECK FAILED: {exc}") from exc
