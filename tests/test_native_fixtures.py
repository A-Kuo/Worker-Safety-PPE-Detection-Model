"""Keep the experimental native port (native/ppe-core) honest from the Python side.

The Rust tests compare against fixtures generated from the Python decode. These tests make sure
those fixtures are current, so changing the Python decode without regenerating them fails here.
The Rust tests themselves run with ``cargo test`` (CI job ``native``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "native" / "ppe-core" / "tests" / "fixtures"


@pytest.fixture(scope="module")
def gen():
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import gen_native_fixtures
    finally:
        sys.path.remove(str(REPO / "scripts"))
    return gen_native_fixtures


def test_committed_fixtures_match_the_current_python_decode(gen):
    generated = gen.generate()
    assert sorted(p.name for p in FIXTURES.glob("*.txt")) == sorted(generated)
    for name, text in generated.items():
        committed = (FIXTURES / name).read_text(encoding="utf-8").replace("\r\n", "\n")
        assert committed == text, (
            f"{name} is out of date: run `python scripts/gen_native_fixtures.py` and re-run `cargo test` "
            "in native/ppe-core"
        )


def test_fixtures_exercise_real_work(gen):
    counts = {}
    for name, text in gen.generate().items():
        counts[name] = int(next(ln for ln in text.splitlines() if ln.startswith("expect ")).split()[1])
    assert counts["nothing_above_threshold.txt"] == 0  # the empty case stays empty
    assert sum(counts.values()) > 100 and sum(1 for v in counts.values() if v > 10) >= 4


@pytest.mark.parametrize("w,h,imgsz", [(1280, 720, 640), (720, 1280, 640), (1024, 683, 640), (96, 96, 64),
                                       (100, 25, 10), (641, 479, 640)])
def test_fixture_geometry_is_the_runtime_letterbox_geometry(gen, w, h, imgsz):
    pytest.importorskip("cv2")
    from ppe.runtime.backends import Letterboxer

    _, meta = Letterboxer(imgsz)(np.zeros((h, w, 3), dtype=np.uint8))
    assert gen.letterbox_meta(w, h, imgsz) == meta


def test_native_crate_declares_no_dependencies_and_is_marked_experimental():
    cargo = (REPO / "native" / "ppe-core" / "Cargo.toml").read_text(encoding="utf-8")
    deps = cargo.split("[dependencies]")[1].split("[")[0]
    assert [ln for ln in deps.splitlines() if ln.strip() and not ln.strip().startswith("#")] == []
    assert "publish = false" in cargo and 'license = "AGPL-3.0-only"' in cargo
    readme = (REPO / "native" / "README.md").read_text(encoding="utf-8")
    assert "experiment" in readme.lower() and "not used by" in readme.lower()
