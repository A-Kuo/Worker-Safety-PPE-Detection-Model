#!/usr/bin/env python3
"""Build the vest specialist dataset. Thin wrapper kept for existing notebooks/docs; the logic (and
the explanation of why a specialist exists) lives in scripts/make_specialist_data.py."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from make_specialist_data import main as _main  # noqa: E402
from make_specialist_data import remap_label_lines as _remap  # noqa: E402

UNIFIED_TO_SPECIALIST = {2: 0, 3: 1}  # vest, no_vest
SPECIALIST_NAMES = ["vest", "no_vest"]


def remap_label_lines(lines) -> list[str]:
    return _remap(lines, UNIFIED_TO_SPECIALIST)


def main() -> int:
    return _main(["--exp", "vest_specialist", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
