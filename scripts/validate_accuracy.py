#!/usr/bin/env python3
"""Validate recorded eval JSONs against the product accuracy bars (no GPU required).

Reads ``results/analysis/eval_*.json`` (or paths you pass) and reports per-class
precision / recall / mAP50 against the README targets:

  * vest / no_vest → 95%+ P and R (priority 1)
  * helmet / no_helmet → high 80s–90s stretch (priority 2)
  * goggles → ~70% acceptable (priority 3)

Understands both eval formats: Ultralytics ``scripts/eval.py`` output and the runtime ``ppe eval``
output, which carries box counts and Wilson intervals. With counts available each line says whether
the sample size can support the bar at all. ``--report`` prints the table and always exits 0.

Exit 0 when every available priority-1 class clears its bar (or when only soft
warnings fire). Exit 1 when a hard bar fails. Use ``--strict`` to also fail on
priority-2/3 misses. This is the offline gate that "validate model accuracy"
should hit before claiming a specialist or threshold change improved anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _common import REPO_ROOT  # noqa: E402


@dataclass(frozen=True)
class Bar:
    precision: float
    recall: float
    map50: float | None = None
    priority: int = 1
    soft: bool = False  # soft bars warn by default; --strict promotes them to fail


# Product bars from README §2. Soft = stretch / acceptable floors that should not
# block a cycle unless --strict.
BARS: dict[str, Bar] = {
    "vest": Bar(0.95, 0.95, priority=1),
    "no_vest": Bar(0.95, 0.95, priority=1),
    "helmet": Bar(0.85, 0.85, priority=2, soft=True),
    "no_helmet": Bar(0.85, 0.85, priority=2, soft=True),
    "goggles": Bar(0.70, 0.70, priority=3, soft=True),
}


# Confidence threshold used to read precision / recall from a runtime eval (`ppe eval`) sweep.
DEFAULT_OPERATING_CONF = 0.30


def load_per_class(path: Path, conf: float = DEFAULT_OPERATING_CONF) -> dict[str, dict]:
    """Per-class rows from either eval format.

    * ``scripts/eval.py`` (Ultralytics val): ``metrics.per_class`` is a list; no box counts.
    * ``ppe eval`` (shipped runtime): ``per_class`` is a dict with ``n_gt_boxes``, ``ap50`` and a
      threshold ``sweep`` carrying counts and Wilson intervals; P/R are read at ``conf``.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") or payload
    rows = metrics.get("per_class") or []
    out: dict[str, dict] = {}
    if isinstance(rows, dict):
        for name, row in rows.items():
            sweep = row.get("sweep") or []
            if not sweep:
                continue
            point = min(sweep, key=lambda s: abs(float(s["conf"]) - conf))
            p, r = point.get("precision"), point.get("recall")
            out[str(name)] = {
                "precision": float(p or 0.0),
                "recall": float(r or 0.0),
                "map50": float(row.get("ap50") or 0.0),
                "f1": (2 * p * r / (p + r)) if p and r else 0.0,
                "n": int(row.get("n_gt_boxes") or 0),
                "conf": float(point["conf"]),
                "precision_ci95": point.get("precision_ci95"),
                "recall_ci95": point.get("recall_ci95"),
            }
        return out
    for row in rows:
        name = str(row.get("name") or "")
        if not name:
            continue
        out[name] = {
            "precision": float(row.get("precision", 0.0)),
            "recall": float(row.get("recall", 0.0)),
            "map50": float(row.get("map50", 0.0)),
            "f1": float(row.get("f1", 0.0)) if row.get("f1") is not None else 0.0,
        }
    return out


def support_note(row: dict, bar: Bar) -> str:
    """What the sample size allows us to claim about a bar (empty when counts are unknown)."""
    n = row.get("n")
    if not n:
        return ""
    lo = [ci[0] for ci in (row.get("precision_ci95"), row.get("recall_ci95")) if ci]
    hi = [ci[1] for ci in (row.get("precision_ci95"), row.get("recall_ci95")) if ci]
    note = f" n={n}"
    if row.get("recall_ci95"):
        note += f" R95%CI=[{row['recall_ci95'][0]:.2f},{row['recall_ci95'][1]:.2f}]"
    if row.get("precision_ci95"):
        note += f" P95%CI=[{row['precision_ci95'][0]:.2f},{row['precision_ci95'][1]:.2f}]"
    if lo and min(lo) >= min(bar.precision, bar.recall):
        note += " (bar supported by the interval)"
    elif hi and min(hi) < min(bar.precision, bar.recall):
        note += " (bar excluded by the interval)"
    else:
        note += " (too few boxes to confirm or exclude the bar)"
    return note


def evaluate(per_class: dict[str, dict], *, strict: bool = False) -> tuple[list[str], list[str], list[str]]:
    """Return (passes, soft_warnings, hard_failures)."""
    passes: list[str] = []
    warnings: list[str] = []
    failures: list[str] = []
    for name, bar in BARS.items():
        row = per_class.get(name)
        if row is None:
            continue
        p, r = row["precision"], row["recall"]
        ok_p, ok_r = p >= bar.precision, r >= bar.recall
        at = f" @conf {row['conf']:.2f}" if "conf" in row else ""
        detail = (
            f"{name}: P={p:.3f} (bar {bar.precision:.2f}) R={r:.3f} (bar {bar.recall:.2f}) "
            f"mAP50={row['map50']:.3f}{at} [P{bar.priority}]{support_note(row, bar)}"
        )
        if ok_p and ok_r:
            passes.append(detail)
            continue
        if bar.soft and not strict:
            warnings.append(detail)
        else:
            failures.append(detail)
    return passes, warnings, failures


def default_eval_paths(root: Path = REPO_ROOT) -> list[Path]:
    analysis = root / "results" / "analysis"
    # Clean (deduplicated) evals only. eval_hexmon_vyra_unified_order.json scores the raw Combined
    # test, where 54% of images have a near-duplicate in train/valid, so it is not a default.
    preferred = [
        analysis / "eval_vest_onnx_clean_npu.json",  # the shipped FP32 ONNX through the runtime, on the NPU
        analysis / "eval_vest_int8_clean_npu.json",
        analysis / "eval_vest_specialist.json",  # Kaggle .pt, Ultralytics val (no box counts)
        analysis / "eval_helmet_specialist.json",
        analysis / "eval_e0_n_clean.json",
        analysis / "eval_hexmon_vyra_clean.json",
    ]
    found = [p for p in preferred if p.is_file()]
    if found:
        return found
    return sorted(analysis.glob("eval_*.json"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("evals", nargs="*", type=Path, help="eval JSON paths (default: known results/analysis/eval_*.json)")
    parser.add_argument("--strict", action="store_true", help="also fail on soft (helmet/goggles) bars")
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    parser.add_argument(
        "--report",
        action="store_true",
        help="print the same table but always exit 0 (for a beta release whose bars are knowingly unmet)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_OPERATING_CONF,
        help="operating confidence at which P/R are read from `ppe eval` sweeps",
    )
    args = parser.parse_args(argv)

    paths = [p if p.is_absolute() else REPO_ROOT / p for p in args.evals] or default_eval_paths()
    if not paths:
        print("No eval JSON files found under results/analysis/", file=sys.stderr)
        return 2

    summary = []
    hard_fail = False
    for path in paths:
        if not path.is_file():
            print(f"MISSING {path}", file=sys.stderr)
            hard_fail = True
            continue
        per_class = load_per_class(path, args.conf)
        passes, warnings, failures = evaluate(per_class, strict=args.strict)
        summary.append(
            {
                "path": str(path),
                "classes_scored": sorted(per_class),
                "passes": passes,
                "warnings": warnings,
                "failures": failures,
            }
        )
        print(f"\n== {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path} ==")
        for line in passes:
            print(f"  PASS  {line}")
        for line in warnings:
            print(f"  WARN  {line}")
        for line in failures:
            print(f"  FAIL  {line}")
            hard_fail = True
        if not passes and not warnings and not failures:
            print("  (no product-bar classes present in this eval)")

    if args.json:
        print(json.dumps({"results": summary, "hard_fail": hard_fail}, indent=2))
    if args.report:
        print("\n(report mode: exit 0 regardless of the bars above)")
        return 0
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
