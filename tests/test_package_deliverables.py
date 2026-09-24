"""Packaging must fail loudly when the run's essentials are missing, and prune only safe roots."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import package_deliverables as pd  # noqa: E402


def _fake_run(root: Path, exp="demo", with_eval=True, with_best=True):
    (root / "runs/train" / exp / "weights").mkdir(parents=True)
    if with_best:
        (root / "runs/train" / exp / "weights/best.pt").write_bytes(b"w" * 100)
    (root / "runs/train" / exp / "results.csv").write_text("epoch\n1\n")
    (root / "runs/train" / exp / "args.yaml").write_text("epochs: 1\n")
    if with_eval:
        (root / "results/analysis").mkdir(parents=True)
        (root / f"results/analysis/eval_{exp}.json").write_text(json.dumps({"metrics": {"map50": 0.8}}))


def test_collect_manifest_and_zip(tmp_path, monkeypatch):
    _fake_run(tmp_path)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    zip_path = pd.run("demo", out)
    names = zipfile.ZipFile(zip_path).namelist()
    assert "weights/best.pt" in names and "eval_demo.json" in names and "MANIFEST.json" in names
    manifest = json.loads((out / "deliverables_demo/MANIFEST.json").read_text())
    assert manifest["metrics"] == {"map50": 0.8}
    assert manifest["files"]["weights/best.pt"]["bytes"] == 100


@pytest.mark.parametrize("kw", [{"with_eval": False}, {"with_best": False}])
def test_missing_essentials_fail_loudly(tmp_path, monkeypatch, kw):
    _fake_run(tmp_path, **kw)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(AssertionError, match="NOT done"):
        pd.run("demo", out)


def test_prune_refuses_arbitrary_directories(tmp_path):
    (tmp_path / "keep_me").mkdir()
    with pytest.raises(ValueError, match="refusing to prune"):
        pd.prune(tmp_path, set())
    assert (tmp_path / "keep_me").exists()


def test_prune_keeps_only_named_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(pd, "PRUNE_ROOTS", (tmp_path,))
    (tmp_path / "ppe/data").mkdir(parents=True)
    (tmp_path / "keep.zip").write_bytes(b"z")
    (tmp_path / "junk.log").write_text("x")
    assert sorted(pd.prune(tmp_path, {"keep.zip"})) == ["junk.log", "ppe"]
    assert [p.name for p in tmp_path.iterdir()] == ["keep.zip"]


def test_run_with_prune_leaves_only_deliverables(tmp_path, monkeypatch):
    work = tmp_path / "work"
    (work / "ppe").mkdir(parents=True)
    _fake_run(work / "ppe")
    (work / "ppe/data").mkdir()
    monkeypatch.chdir(work / "ppe")
    monkeypatch.setattr(pd, "PRUNE_ROOTS", (work,))
    pd.run("demo", work, do_prune=True)
    assert sorted(p.name for p in work.iterdir()) == ["deliverables_demo", "deliverables_demo.zip"]


def test_auto_package_is_a_noop_off_kaggle_and_for_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(pd, "KAGGLE_WORKING", tmp_path / "not_kaggle")
    assert pd.auto_package("demo", tmp_path, prune_after=True, strict=True) is None
    (tmp_path / "kaggle").mkdir()
    monkeypatch.setattr(pd, "KAGGLE_WORKING", tmp_path / "kaggle")
    assert pd.auto_package("smoke", tmp_path, prune_after=True, strict=True) is None
    monkeypatch.setenv("PPE_NO_PACKAGE", "1")
    assert pd.auto_package("demo", tmp_path, prune_after=True, strict=True) is None
    assert list((tmp_path / "kaggle").iterdir()) == []


def test_auto_package_on_kaggle_stages_then_prunes(tmp_path, monkeypatch):
    work = tmp_path / "kaggle"
    repo = work / "ppe"
    repo.mkdir(parents=True)
    _fake_run(repo)
    (repo / "data").mkdir()
    (work / "train_demo.log").write_text("log")
    monkeypatch.chdir(repo)
    monkeypatch.setattr(pd, "KAGGLE_WORKING", work)
    monkeypatch.setattr(pd, "PRUNE_ROOTS", (work,))
    # train.py-style call: non-strict, no prune, eval JSON not there yet
    (repo / "results/analysis/eval_demo.json").unlink()
    pd.auto_package("demo", repo, prune_after=False, strict=False)
    assert (work / "deliverables_demo/weights/best.pt").exists() and (repo / "data").exists()
    # sanity-style call: strict, prune
    (repo / "results/analysis/eval_demo.json").write_text(json.dumps({"metrics": {"map50": 0.8}}))
    pd.auto_package("demo", repo, prune_after=True, strict=True)
    assert sorted(p.name for p in work.iterdir()) == ["deliverables_demo", "deliverables_demo.zip", "train_demo.log"]
