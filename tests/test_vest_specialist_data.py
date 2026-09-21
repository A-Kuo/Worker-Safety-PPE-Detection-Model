"""The specialist dataset must keep only vest/no_vest, renumbered 0/1, and never touch other classes."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import make_vest_specialist_data as msd  # noqa: E402


def test_remap_keeps_only_vest_and_no_vest_renumbered():
    lines = [
        "0 0.5 0.5 0.1 0.1",   # helmet  -> dropped
        "2 0.4 0.4 0.2 0.3",   # vest    -> 0
        "3 0.6 0.6 0.2 0.3",   # no_vest -> 1
        "10 0.5 0.5 0.9 0.9",  # person  -> dropped
    ]
    assert msd.remap_label_lines(lines) == ["0 0.4 0.4 0.2 0.3", "1 0.6 0.6 0.2 0.3"]


def test_remap_ignores_blank_and_malformed_lines():
    assert msd.remap_label_lines(["", "   ", "3 0.1", "x 1 2 3 4", "3 0.5 0.5 0.1 0.1"]) == ["1 0.5 0.5 0.1 0.1"]


def test_specialist_names_match_unified_class_names():
    from ppe.schema import UNIFIED_CLASS_NAMES

    # Inference maps model class names to unified names by string, so these must be unified names.
    assert all(name in UNIFIED_CLASS_NAMES for name in msd.SPECIALIST_NAMES)
    assert [UNIFIED_CLASS_NAMES[i] for i in msd.UNIFIED_TO_SPECIALIST] == msd.SPECIALIST_NAMES
