"""Offline product-bar validation against recorded eval JSONs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import validate_accuracy as va  # noqa: E402


def _write_eval(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        json.dumps({"metrics": {"precision": 0.5, "recall": 0.5, "map50": 0.5, "per_class": rows}}),
        encoding="utf-8",
    )
    return path


def test_vest_bars_pass_and_fail(tmp_path):
    good = {
        "name": "vest",
        "precision": 0.96,
        "recall": 0.95,
        "map50": 0.9,
        "f1": 0.95,
    }
    bad = {"name": "no_vest", "precision": 0.67, "recall": 0.72, "map50": 0.75, "f1": 0.7}
    per = va.load_per_class(_write_eval(tmp_path / "e.json", [good, bad]))
    passes, warnings, failures = va.evaluate(per)
    assert any(p.startswith("vest:") for p in passes)
    assert any(f.startswith("no_vest:") for f in failures)
    assert warnings == []


def test_soft_helmet_bar_warns_unless_strict():
    per = {
        "helmet": {"precision": 0.80, "recall": 0.80, "map50": 0.75, "f1": 0.8},
        "no_helmet": {"precision": 0.90, "recall": 0.90, "map50": 0.85, "f1": 0.9},
    }
    _, warnings, failures = va.evaluate(per, strict=False)
    assert failures == []
    assert any("helmet:" in w for w in warnings)
    _, _, hard = va.evaluate(per, strict=True)
    assert any("helmet:" in f for f in hard)


def test_cli_reports_existing_vest_specialist_eval():
    path = ROOT / "results" / "analysis" / "eval_vest_specialist.json"
    if not path.is_file():
        pytest.skip("vest specialist eval artifact not in tree")
    # Current vest specialist is below the 95% product bar on no_vest — must surface FAIL.
    code = va.main([str(path)])
    assert code == 1
