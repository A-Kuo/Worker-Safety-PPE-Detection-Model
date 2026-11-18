"""Model registry, verified download, generated detector and release checks (PR 4)."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ppe import models

REPO = Path(__file__).resolve().parents[1]


def _entry(payload: bytes, name="m", file="m.onnx"):
    return models.ModelEntry(name=name, file=file, url="https://example.invalid/m.onnx",
                             sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload), conf=0.3)


def _opener(payload: bytes, chunks=None):
    calls = []

    def open_url(request, timeout=None):
        calls.append(request.full_url)
        return io.BytesIO(payload)

    open_url.calls = calls
    return open_url


# --- registry -------------------------------------------------------------------------------------

def test_registry_is_packaged_and_well_formed():
    table = models.entries()
    assert models.default_name() in table and "vest_specialist_int8" in table
    fp32 = models.get_entry()
    assert fp32.file == "vest_specialist.onnx" and fp32.bytes == 12_239_302 and fp32.conf == 0.30
    assert fp32.sha256 == "b641539a7c56036382e9418b3df7a3b3f107f62f2059b65f4d8356e77e1c96b4"
    assert len({e.sha256 for e in table.values()}) == len(table)
    with pytest.raises(KeyError, match="Unknown model"):
        models.get_entry("nope")


def test_model_is_recognised_by_hash_not_by_name():
    fp32 = models.get_entry("vest_specialist")
    assert models.recommended_conf(fp32.sha256) == 0.30
    assert models.recommended_conf(fp32.sha256.upper()) == 0.30
    assert models.recommended_conf("0" * 64) is None and models.recommended_conf(None) is None


def test_release_check_passes_for_the_repo_and_catches_mismatches(tmp_path):
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import check_release
    finally:
        sys.path.remove(str(REPO / "scripts"))

    assert check_release.problems() == []
    version = check_release.package_version()
    assert check_release.problems(tag=f"v{version}") == []
    assert any("tag" in p for p in check_release.problems(tag="v9.9.9"))

    # A copy with a wrong URL, an undocumented hash and a bad default must be reported.
    (tmp_path / "src" / "ppe").mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "pyproject.toml").write_text('version = "0.1.0"\n')
    (tmp_path / "docs" / "MODEL_CARD.md").write_text("no hashes here")
    bad = {"release": "v0.2.0", "default": "missing", "models": {"m": {
        "file": "m.onnx", "url": "http://x/releases/download/v0.1.0/m.onnx", "sha256": "ABC", "bytes": 0, "conf": 1.5}}}
    (tmp_path / "src" / "ppe" / "models.json").write_text(json.dumps(bad))
    found = "\n".join(check_release.problems(tmp_path))
    for needle in ("release 'v0.2.0'", "default 'missing'", "sha256 is not 64", "bytes must be", "url must be https",
                   "conf 1.5", "not documented"):
        assert needle in found, (needle, found)


# --- fetch ------------------------------------------------------------------------------------------

def test_fetch_downloads_verifies_and_installs_atomically(tmp_path):
    payload = b"model-bytes" * 1000
    entry = _entry(payload)
    opener = _opener(payload)
    seen = []
    path, downloaded = models.fetch(entry, tmp_path / "store", opener=opener, progress=lambda a, b: seen.append((a, b)))
    assert downloaded and path == tmp_path / "store" / "m.onnx" and path.read_bytes() == payload
    assert opener.calls == [entry.url] and seen[-1] == (len(payload), len(payload))
    assert list((tmp_path / "store").glob("*.part")) == []
    assert models.is_verified(path, entry)


def test_fetch_skips_a_verified_copy_and_force_redownloads(tmp_path):
    payload = b"abc" * 50
    entry = _entry(payload)
    models.fetch(entry, tmp_path, opener=_opener(payload))
    opener = _opener(payload)
    _, downloaded = models.fetch(entry, tmp_path, opener=opener)
    assert downloaded is False and opener.calls == []  # no network when the file is already right
    _, downloaded = models.fetch(entry, tmp_path, opener=opener, force=True)
    assert downloaded is True and len(opener.calls) == 1


def test_fetch_replaces_a_corrupt_existing_file(tmp_path):
    payload = b"good" * 100
    entry = _entry(payload)
    (tmp_path / "m.onnx").write_bytes(b"bad!" * 100)  # same size, wrong content
    path, downloaded = models.fetch(entry, tmp_path, opener=_opener(payload))
    assert downloaded and path.read_bytes() == payload


@pytest.mark.parametrize(
    "served, message",
    [(b"x" * 400, "SHA-256 mismatch"), (b"short", "got 5 bytes"), (b"y" * 5000, "larger than the registered")],
)
def test_fetch_rejects_wrong_content_and_leaves_nothing_behind(tmp_path, served, message):
    entry = _entry(b"z" * 400)
    with pytest.raises(models.ModelVerificationError, match=message):
        models.fetch(entry, tmp_path, opener=_opener(served))
    assert list(tmp_path.iterdir()) == []  # neither the model nor a partial file


def test_failed_fetch_keeps_an_existing_good_file_untouched_until_verified(tmp_path):
    payload = b"keep" * 100
    entry = _entry(payload)
    models.fetch(entry, tmp_path, opener=_opener(payload))
    with pytest.raises(models.ModelVerificationError):
        models.fetch(entry, tmp_path, opener=_opener(b"evil" * 100), force=True)
    assert (tmp_path / "m.onnx").read_bytes() == payload  # the bad download never replaced it


def test_fetch_rejects_unknown_url_schemes(tmp_path):
    with pytest.raises(ValueError, match="Unsupported URL scheme"):
        models.fetch(_entry(b"x"), tmp_path, url="ftp://host/m.onnx", opener=_opener(b"x"))


def test_fetch_from_a_real_local_file_url(tmp_path):
    payload = b"local" * 200
    src = tmp_path / "src.bin"
    src.write_bytes(payload)
    path, downloaded = models.fetch(_entry(payload), tmp_path / "dest", url=src.as_uri())
    assert downloaded and path.read_bytes() == payload


# --- CLI --------------------------------------------------------------------------------------------

def test_cli_fetch_model_list_and_errors(tmp_path, capsys, monkeypatch):
    from ppe import cli

    assert cli.main(["fetch-model", "--list"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["name"] for r in rows if r["default"]] == ["vest_specialist"]
    assert all(len(r["sha256"]) == 64 for r in rows)

    with pytest.raises(SystemExit, match="Unknown model"):
        cli.main(["fetch-model", "nope", "--dest", str(tmp_path)])
    with pytest.raises(SystemExit, match="--url applies to a single model"):
        cli.main(["fetch-model", "--all", "--url", "file:///x", "--dest", str(tmp_path)])

    fake = tmp_path / "fake.onnx"
    fake.write_bytes(b"not the model")
    with pytest.raises(SystemExit, match="expected 12239302"):
        cli.main(["fetch-model", "--dest", str(tmp_path / "d"), "--url", fake.as_uri()])
    assert not (tmp_path / "d" / "vest_specialist.onnx").exists()

    with pytest.raises(SystemExit, match="could not download"):
        cli.main(["fetch-model", "--dest", str(tmp_path / "d"), "--url", (tmp_path / "missing.onnx").as_uri()])


def test_cli_fetch_model_installs_into_the_model_dir(tmp_path, monkeypatch, capsys):
    from ppe import cli

    payload = b"tiny" * 10
    entry = replace(models.get_entry(), sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
    monkeypatch.setattr(models, "entries", lambda: {entry.name: entry})
    monkeypatch.setenv("PPE_MODEL_DIR", str(tmp_path / "store"))
    src = tmp_path / "asset.onnx"
    src.write_bytes(payload)
    assert cli.main(["fetch-model", "--url", src.as_uri()]) == 0
    assert "downloaded and verified" in capsys.readouterr().out
    assert (tmp_path / "store" / "vest_specialist.onnx").read_bytes() == payload
    assert cli.main(["fetch-model", "--url", src.as_uri()]) == 0
    assert "already present, verified" in capsys.readouterr().out


class _Args:
    def __init__(self, conf=None):
        self.conf = conf


class _Sess:
    def __init__(self, sha):
        from ppe.runtime.config import ExecutionPolicy

        self.policy = ExecutionPolicy()
        self.backend = type("B", (), {"fingerprint": {"model": {"sha256": sha}}})()


def test_registered_model_gets_its_recommended_threshold_unless_the_user_chose(monkeypatch):
    from ppe import cli

    sha = models.get_entry().sha256
    monkeypatch.delenv("PPE_CONF", raising=False)
    s = _Sess(sha)
    cli._apply_model_defaults(s, _Args())
    assert s.policy.conf == 0.30
    s = _Sess("f" * 64)  # unknown model: package default untouched
    cli._apply_model_defaults(s, _Args())
    assert s.policy.conf == 0.25
    s = _Sess(sha)
    s.policy.conf = 0.5
    cli._apply_model_defaults(s, _Args(conf=0.5))  # explicit flag wins
    assert s.policy.conf == 0.5
    monkeypatch.setenv("PPE_CONF", "0.4")
    s = _Sess(sha)
    s.policy.conf = 0.4
    cli._apply_model_defaults(s, _Args())  # environment wins
    assert s.policy.conf == 0.4


def test_conf_flag_absent_respects_ppe_conf_env(monkeypatch):
    from ppe import cli

    monkeypatch.setenv("PPE_CONF", "0.41")
    args = cli.build_parser().parse_args(["predict", "--source", "x.jpg"])
    assert args.conf is None and cli._policy_from_args(args).conf == 0.41
    args = cli.build_parser().parse_args(["predict", "--source", "x.jpg", "--conf", "0.6"])
    assert cli._policy_from_args(args).conf == 0.6


# --- generated detector + smoke scripts -----------------------------------------------------------

def test_tiny_detector_runs_through_the_real_runtime_with_exact_boxes(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from ppe.runtime import ExecutionPolicy, open_session
    from ppe.testing import make_tiny_detector

    path = make_tiny_detector(tmp_path / "t.onnx")
    session = open_session(path, policy=ExecutionPolicy(imgsz=64, providers=("cpu",)), repo_root=tmp_path)
    dets = sorted(session.predict(np.full((96, 128, 3), 114, dtype=np.uint8)), key=lambda d: -d.conf)
    assert [(d.cls_name, round(d.conf, 2)) for d in dets] == [("vest", 0.9), ("no_vest", 0.8)]
    assert dets[0].xyxy == pytest.approx((24.0, 8.0, 56.0, 40.0), abs=0.01)  # letterbox undone exactly
    assert session.backend.class_names == {0: "vest", 1: "no_vest"}
    assert session.backend.static_input is True
    # Image content does not change the output (the model is a fixture, not a detector).
    noisy = np.random.default_rng(0).integers(0, 255, (96, 128, 3), dtype=np.uint8)
    assert len(session.predict(noisy)) == 2


def test_tiny_detector_supports_custom_classes_and_validates_arguments(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from ppe.runtime import ExecutionPolicy, open_session
    from ppe.testing import make_tiny_detector

    path = make_tiny_detector(tmp_path / "p.onnx", names=("person", "helmet", "vest"),
                              boxes=[(0, 0.7, 32, 32, 20, 40)])
    session = open_session(path, policy=ExecutionPolicy(imgsz=64, providers=("cpu",)), repo_root=tmp_path)
    assert [d.cls_name for d in session.predict(np.zeros((64, 64, 3), dtype=np.uint8))] == ["person"]
    with pytest.raises(ValueError, match="anchors must exceed"):
        make_tiny_detector(tmp_path / "x.onnx", anchors=6)


def test_edge_smoke_runs_without_torch_or_tracked_weights():
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    proc = subprocess.run([sys.executable, str(REPO / "scripts" / "smoke_edge.py"), "--require-npu"],
                          capture_output=True, text=True, cwd=REPO, timeout=300)
    assert proc.returncode == 0, proc.stdout[-800:] + proc.stderr[-800:]
    assert "EDGE_SMOKE_OK" in proc.stdout and "HARD_NPU_GATE" in proc.stdout
    src = (REPO / "scripts" / "smoke_edge.py").read_text(encoding="utf-8")
    assert "baselines" not in src  # no dependency on archived third-party weights


def test_opening_a_session_does_not_import_torch_even_when_torch_is_installed(tmp_path):
    """Regression: the Windows DLL helper used to import ``onnxruntime.tools``, whose package init
    imports torch when present. Checked in a subprocess so other tests' imports cannot mask it."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    code = (
        "import sys, numpy as np\n"
        "from pathlib import Path\n"
        "from ppe.testing import make_tiny_detector\n"
        "from ppe.runtime import ExecutionPolicy, open_session\n"
        f"p = make_tiny_detector(Path(r'{tmp_path}') / 't.onnx')\n"
        f"s = open_session(p, policy=ExecutionPolicy(imgsz=64), repo_root=Path(r'{tmp_path}'))\n"
        "s.predict(np.zeros((64, 64, 3), np.uint8)); s.info()\n"
        "bad = [m for m in ('torch', 'ultralytics', 'onnxruntime.tools') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    import os

    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO,
                          env={**os.environ, "PYTHONPATH": str(REPO / "src")}, timeout=300)
    assert proc.returncode == 0, proc.stderr[-800:]


def test_workflows_are_valid_yaml_and_use_the_torch_free_path():
    yaml = pytest.importorskip("yaml")
    wf = REPO / ".github" / "workflows"
    ci = yaml.safe_load((wf / "ci.yml").read_text(encoding="utf-8"))
    assert set(ci["jobs"]) == {"runtime", "gui", "train", "native"}
    matrix = ci["jobs"]["runtime"]["strategy"]["matrix"]
    assert matrix["python-version"] == ["3.10", "3.11", "3.12"]
    assert {"os": "windows-latest", "python-version": "3.12"} in matrix["include"]
    runtime_steps = " ".join(str(s.get("run", "")) for s in ci["jobs"]["runtime"]["steps"])
    assert '.[cpu,dev]' in runtime_steps and "requirements.txt" not in runtime_steps
    for needed in ("pytest", "smoke_edge.py", "python -m build", "twine check", "wheel_smoke.py"):
        assert needed in runtime_steps, needed
    release = yaml.safe_load((wf / "release.yml").read_text(encoding="utf-8"))
    release_steps = " ".join(str(s.get("run", "")) for s in release["jobs"]["build"]["steps"])
    assert "--draft" in release_steps and "check_release.py" in release_steps  # never auto-publishes
    post = yaml.safe_load((wf / "post-release.yml").read_text(encoding="utf-8"))
    assert "workflow_dispatch" in post[True] if True in post else "workflow_dispatch" in post["on"]
