"""Licence, notices and documentation stay true (PR 5). No network, dataset or hardware needed."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
# SHA-256 of the canonical GNU AGPL v3 text (https://www.gnu.org/licenses/agpl-3.0.txt), LF endings.
AGPL3_SHA256 = "0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0"
MEDIA_OR_WEIGHTS = {".jpg", ".jpeg", ".jfif", ".png", ".gif", ".webp", ".mp4", ".avi", ".mov", ".pt", ".pth", ".onnx"}


def _text(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


# --- licence ----------------------------------------------------------------------------------------

def test_license_file_is_the_unmodified_agpl3_text():
    raw = (REPO / "LICENSE").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == AGPL3_SHA256
    assert raw.lstrip().startswith(b"GNU AFFERO GENERAL PUBLIC LICENSE")


def test_licence_is_declared_consistently_everywhere():
    tomllib = pytest.importorskip("tomllib")
    project = tomllib.loads(_text("pyproject.toml"))["project"]
    assert project["license"] == {"text": "AGPL-3.0-only"}
    assert "License :: OSI Approved :: GNU Affero General Public License v3" in project["classifiers"]
    assert "Development Status :: 4 - Beta" in project["classifiers"]
    assert project["urls"]["Source"].startswith("https://github.com/")

    from ppe import about

    assert about.LICENSE == "AGPL-3.0-only"
    assert project["urls"]["Source"] == about.SOURCE_URL
    registry = json.loads(_text("src/ppe/models.json"))
    assert {m["license"] for m in registry["models"].values()} == {"AGPL-3.0-only"}
    for rel in ("README.md", "NOTICE", "ATTRIBUTION.md", "CONTRIBUTING.md"):
        assert "AGPL-3.0" in _text(rel) or "Affero General Public License" in _text(rel), rel
    notice = _text("NOTICE")
    assert "version 3." in notice and "any later version" not in notice  # "-only", not "-or-later"
    assert about.SOURCE_URL in notice and "Ultralytics" in notice


def test_required_project_files_exist():
    for rel in ("LICENSE", "NOTICE", "ATTRIBUTION.md", "CHANGELOG.md", "CONTRIBUTING.md", "SECURITY.md",
                "docs/README.md", "docs/MODEL_CARD.md", "docs/RELEASING.md"):
        assert (REPO / rel).is_file(), rel


# --- third-party material ---------------------------------------------------------------------------

def test_no_unlicensed_media_or_weights_remain_in_the_archive():
    leftovers = [str(p.relative_to(REPO)) for p in (REPO / "archive").rglob("*")
                 if p.is_file() and p.suffix.lower() in MEDIA_OR_WEIGHTS]
    assert leftovers == []


def test_the_only_committed_image_is_our_own_screenshot():
    tracked_dirs = ("docs", "src", "scripts", "tests", "configs", "app", "archive")
    images = sorted(
        str(p.relative_to(REPO)).replace("\\", "/")
        for d in tracked_dirs
        for p in (REPO / d).rglob("*")
        if p.is_file() and p.suffix.lower() in MEDIA_OR_WEIGHTS and "__pycache__" not in p.parts
    )
    assert images == ["docs/assets/window_hud.png"]


def test_nothing_runs_the_removed_baseline_by_default():
    pattern = re.compile(r'"archive"\s*/\s*"baselines"\s*/[^\n]*"models"')
    offenders = [str(p.relative_to(REPO)) for base in ("src", "app") for p in (REPO / base).rglob("*.py")
                 if pattern.search(p.read_text(encoding="utf-8", errors="ignore"))]
    assert offenders == []
    assert "archive/baselines" not in _text(".gitignore")


def test_default_model_lookup_no_longer_falls_back_to_the_baseline(tmp_path, monkeypatch):
    from ppe.runtime.config import resolve_model_path

    fake = tmp_path / "archive" / "baselines" / "snehilsanyal_yolov8n_css" / "models"
    fake.mkdir(parents=True)
    (fake / "best.onnx").write_bytes(b"x")
    (fake / "best.pt").write_bytes(b"x")
    monkeypatch.setenv("PPE_MODEL_DIR", str(tmp_path / "none"))
    with pytest.raises(FileNotFoundError, match="ppe fetch-model"):
        resolve_model_path(None, repo_root=tmp_path)


def test_demo_finds_the_fetched_model(tmp_path, monkeypatch):
    import importlib

    store = tmp_path / "store"
    store.mkdir()
    (store / "vest_specialist.onnx").write_bytes(b"x")
    monkeypatch.setenv("PPE_MODEL_DIR", str(store))
    monkeypatch.delenv("PPE_WEIGHTS", raising=False)
    import app.paths as paths

    paths = importlib.reload(paths)
    try:
        monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
        assert store / "vest_specialist.onnx" in paths.DEFAULT_WEIGHT_CANDIDATES
    finally:
        monkeypatch.undo()
        importlib.reload(paths)


# --- README honesty ---------------------------------------------------------------------------------

def test_readme_states_beta_status_and_has_no_stale_claims():
    readme = _text("README.md")
    assert "Beta" in readme and "95%" in readme and "not met" in readme
    assert "not a safety system" in readme
    for stale in ("pending", "0.0.0.0", "requirements.txt\npython -m pip install -r app", "14-class schema"):
        assert stale not in readme, stale
    assert "ppe fetch-model" in readme and "ppe verify-npu" in readme and "ppe window" in readme


def test_readme_accuracy_table_matches_the_committed_evidence():
    evidence = json.loads(_text("results/analysis/eval_vest_onnx_clean_npu.json"))
    readme = _text("README.md")
    for cls in ("vest", "no_vest"):
        pc = evidence["per_class"][cls]
        point = next(s for s in pc["sweep"] if s["conf"] == 0.30)
        cells = [
            str(pc["n_gt_boxes"]),
            f"{pc['ap50']:.3f}",
            f"{point['precision']:.3f} ({point['precision_ci95'][0]:.3f} – {point['precision_ci95'][1]:.3f})",
            f"{point['recall']:.3f} ({point['recall_ci95'][0]:.3f} – {point['recall_ci95'][1]:.3f})",
        ]
        row = f"| {cls} | " + " | ".join(cells) + " |"
        assert row in readme, f"README row for {cls} should be:\n{row}"


def test_readme_latency_table_matches_the_committed_summary():
    summary = json.loads(_text("results/bench/vest_latency_summary.json"))["configs"]
    readme = _text("README.md")

    def span(values):
        lo, hi = values
        return f"{lo:.2f} – {hi:.2f}"

    for key, label in (("int8_npu", "| INT8 | NPU, strict |"), ("fp32_igpu", "| FP32 | iGPU |"),
                       ("fp32_npu", "| FP32 | NPU, strict |")):
        row = f"{label} {span(summary[key]['p50_ms'])} | {span(summary[key]['p95_ms'])} |"
        assert row in readme, row
    cpu = summary["fp32_cpu"]
    assert f"| FP32 | CPU (OpenVINO) | {cpu['p50_ms'][0]:.1f} | {cpu['p95_ms'][0]:.1f} |" in readme


def test_model_hashes_in_the_model_card_match_the_registry():
    card = _text("docs/MODEL_CARD.md")
    for m in json.loads(_text("src/ppe/models.json"))["models"].values():
        assert m["sha256"] in card and f"{m['bytes']:,}" in card


# --- links ------------------------------------------------------------------------------------------

LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
DOCS = ["README.md", "CONTRIBUTING.md", "SECURITY.md", "CHANGELOG.md", "ATTRIBUTION.md", "docs/README.md",
        "docs/MODEL_CARD.md", "docs/RELEASING.md", "docs/edge_npu.md", "docs/experiments.md", "archive/README.md"]


@pytest.mark.parametrize("rel", DOCS)
def test_relative_links_point_at_files_that_exist(rel):
    base = (REPO / rel).parent
    missing = []
    for target in LINK.findall(_text(rel)):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        path = (base / target.split("#")[0]).resolve()
        if not path.exists():
            missing.append(target)
    assert missing == [], f"{rel}: broken links {missing}"
