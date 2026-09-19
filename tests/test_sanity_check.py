"""The post-training sanity verdict must separate 'mislabeled' from 'short run'."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import sanity_check  # noqa: E402


def test_mislabeled_run_fails():
    # The real E0 incident: val 0.769 from training, 0.049 on independent eval.
    status, _ = sanity_check.verdict(best_val=0.769, test=0.049)
    assert status == "fail"


def test_healthy_run_passes():
    assert sanity_check.verdict(best_val=0.769, test=0.74)[0] == "pass"


def test_short_run_is_a_warning_not_a_failure():
    # A 1-2 epoch smoke run: both numbers are tiny and noisy; must not false-alarm.
    assert sanity_check.verdict(best_val=0.004, test=0.0005)[0] == "warn"
    assert sanity_check.verdict(best_val=0.0, test=0.0)[0] == "warn"


def test_undertrained_but_consistent_warns():
    assert sanity_check.verdict(best_val=0.25, test=0.22)[0] == "warn"


def test_read_best_val_map50_handles_padded_headers(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "epoch,   metrics/mAP50(B),  val/box_loss\n1,0.50,1.6\n2,0.52,1.7\n3,0.48,1.8\n",
        encoding="utf-8",
    )
    assert sanity_check.read_best_val_map50(csv_path) == (0.52, 3)


def test_read_best_val_map50_rejects_unexpected_schema(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("epoch,loss\n1,0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mAP50"):
        sanity_check.read_best_val_map50(csv_path)


def test_run_reports_a_missing_checkpoint(tmp_path):
    with pytest.raises(AssertionError, match="did not produce a checkpoint"):
        sanity_check.run("nope", project=tmp_path)
