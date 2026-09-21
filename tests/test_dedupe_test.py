"""dedupe_test must stay fast on numpy < 2 (Kaggle) and must actually recognise duplicates."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import dedupe_test  # noqa: E402


def _reference_popcount(values) -> list[int]:
    return [bin(int(v)).count("1") for v in values]


def test_popcount_matches_reference_including_edge_values():
    values = np.array([0, 1, 2**63, 2**64 - 1, 0xF0F0F0F0F0F0F0F0, 12345678901234567890], dtype=np.uint64)
    assert dedupe_test._popcount(values).tolist() == _reference_popcount(values)


def test_popcount_fallback_without_bitwise_count_is_correct_and_vectorized(monkeypatch):
    monkeypatch.delattr(np, "bitwise_count", raising=False)  # simulate numpy < 2
    rng = np.random.default_rng(0)
    values = rng.integers(0, 2**63, size=200_000, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    start = time.perf_counter()
    got = dedupe_test._popcount(values)
    elapsed = time.perf_counter() - start
    assert got[:1000].tolist() == _reference_popcount(values[:1000])
    assert elapsed < 2.0, f"fallback popcount took {elapsed:.1f}s for 200k values - not vectorized?"


def test_dhash_identical_images_match_and_different_images_do_not(tmp_path):
    cv2 = pytest.importorskip("cv2")
    gradient = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    reversed_gradient = gradient[:, ::-1].copy()
    a, b, c = tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"
    cv2.imwrite(str(a), gradient)
    cv2.imwrite(str(b), gradient)
    cv2.imwrite(str(c), reversed_gradient)
    ha, hb, hc = (dedupe_test.dhash(p) for p in (a, b, c))
    assert ha == hb
    assert bin(ha ^ hc).count("1") > 6  # well beyond the near-duplicate threshold
    assert dedupe_test.dhash(tmp_path / "missing.png") is None
