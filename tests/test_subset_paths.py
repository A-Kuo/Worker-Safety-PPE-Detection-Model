"""Regression tests for the subset-list / symlink label bug.

On Linux (Kaggle), scripts/remap_labels.py builds data/processed/*/images as
symlinks into data/raw/*/images. make_subset.py used Path.resolve() on each
image, so train.txt listed *raw* paths; Ultralytics derives labels by swapping
/images/ -> /labels/, so training silently read the raw, unremapped labels.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import make_subset  # noqa: E402
import train  # noqa: E402

from ppe.schema import UNIFIED_CLASS_NAMES  # noqa: E402


def _norm(path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _symlinks_supported(tmp_path: Path) -> bool:
    target = tmp_path / "_probe_target"
    target.write_text("x")
    link = tmp_path / "_probe_link"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        return False
    return True


def _build_symlinked_dataset(tmp_path: Path, n: int = 6) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    (raw / "train" / "images").mkdir(parents=True)
    (raw / "train" / "labels").mkdir(parents=True)
    (processed / "train" / "images").mkdir(parents=True)
    (processed / "train" / "labels").mkdir(parents=True)
    for i in range(n):
        raw_img = raw / "train" / "images" / f"img{i}.jpg"
        raw_img.write_bytes(b"\xff\xd8\xff\xd9")
        # raw id 3 == Roboflow "Hardhat"; remapped id 0 == unified "helmet".
        (raw / "train" / "labels" / f"img{i}.txt").write_text("3 0.5 0.5 0.2 0.2\n")
        (processed / "train" / "images" / f"img{i}.jpg").symlink_to(raw_img)
        (processed / "train" / "labels" / f"img{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    names = "\n".join(f"  - {name}" for name in UNIFIED_CLASS_NAMES)
    (processed / "data.yaml").write_text(
        f"train: train/images\nval: train/images\nnc: {len(UNIFIED_CLASS_NAMES)}\nnames:\n{names}\n",
        encoding="utf-8",
    )
    return raw, processed


def test_make_subset_keeps_processed_paths_when_images_are_symlinks(tmp_path, monkeypatch):
    if not _symlinks_supported(tmp_path):
        pytest.skip("symlinks unavailable on this OS/account (Linux CI exercises this)")
    raw, processed = _build_symlinked_dataset(tmp_path)
    out = tmp_path / "subset"
    monkeypatch.setattr(
        sys, "argv",
        ["make_subset.py", "--source", str(processed), "--out", str(out), "--n", "6", "--seed", "1"],
    )
    assert make_subset.main() == 0

    lines = [ln for ln in (out / "train.txt").read_text().splitlines() if ln.strip()]
    assert len(lines) == 6
    for line in lines:
        assert _norm(line).startswith(_norm(processed) + os.sep), f"{line} escaped into raw/"
        assert not _norm(line).startswith(_norm(raw) + os.sep)
        label = make_subset.label_path_for_image(Path(line))
        assert label.read_text().split()[0] == "0", "subset must read the remapped label (helmet=0), not raw (3)"


def _write_subset(dest: Path, source: Path, entries: list[str]) -> Path:
    dest.mkdir(parents=True)
    (dest / "train.txt").write_text("\n".join(entries) + "\n")
    (dest / "subset_manifest.json").write_text(json.dumps({"source": str(source)}))
    (dest / "data.yaml").write_text("train: train.txt\nval: train.txt\nnc: 14\nnames: []\n")
    return dest / "data.yaml"


def test_verify_subset_paths_accepts_lists_inside_recorded_source(tmp_path):
    source = tmp_path / "processed"
    yaml_path = _write_subset(
        tmp_path / "subset", source, [str(source / "train" / "images" / "a.jpg").replace("\\", "/")]
    )
    train.verify_subset_paths(yaml_path)


def test_verify_subset_paths_rejects_lists_that_escape_the_recorded_source(tmp_path):
    source = tmp_path / "processed"
    raw = tmp_path / "raw"
    yaml_path = _write_subset(
        tmp_path / "subset", source, [str(raw / "train" / "images" / "a.jpg").replace("\\", "/")]
    )
    with pytest.raises(SystemExit, match="outside the subset's recorded source"):
        train.verify_subset_paths(yaml_path)
