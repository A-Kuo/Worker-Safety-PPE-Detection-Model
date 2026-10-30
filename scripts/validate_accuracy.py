#!/usr/bin/env python3
"""Validate recorded eval JSONs against the product accuracy bars (no GPU required).

Reads ``results/analysis/eval_*.json`` (or paths you pass) and reports per-class
precision / recall / mAP50 against the README targets:

  * vest / no_vest → 95%+ P and R (priority 1)
  * helmet / no_helmet → high 80s–90s stretch (priority 2)
  * goggles → ~70% acceptable (priority 3)

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


def load_per_class(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") or payload
    rows = metrics.get("per_class") or []
    out: dict[str, dict[str, float]] = {}
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


def evaluate(per_class: dict[str, dict[str, float]], *, strict: bool = False) -> tuple[list[str], list[str], list[str]]:
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
        detail = (
            f"{name}: P={p:.3f} (bar {bar.precision:.2f}) R={r:.3f} (bar {bar.recall:.2f}) "
            f"mAP50={row['map50']:.3f} [P{bar.priority}]"
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
    preferred = [
        analysis / "eval_vest_specialist.json",
        analysis / "eval_vest_specialist_local.json",
        analysis / "eval_helmet_specialist.json",
        analysis / "eval_e0_n_clean.json",
        analysis / "eval_hexmon_vyra_unified_order.json",
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
        per_class = load_per_class(path)
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
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
