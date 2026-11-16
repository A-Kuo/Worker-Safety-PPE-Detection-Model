"""Regression tests for the post-merge review findings (strict mode, CLI, archive, smoke)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest

from ppe.runtime.config import ExecutionPolicy

REPO = Path(__file__).resolve().parents[1]
OV = "OpenVINOExecutionProvider"
CPU = "CPUExecutionProvider"


def _tiny_model(tmp_path, shape=(1, 3, 32, 32)):
    pytest.importorskip("onnx")
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("images", TensorProto.FLOAT, list(shape))
    y = helper.make_tensor_value_info("out", TensorProto.FLOAT, list(shape))
    graph = helper.make_graph([helper.make_node("Relu", ["images"], ["out"], name="relu0")], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "tiny.onnx"
    onnx.save(model, str(path))
    return path


# --- app status honors strict mode -------------------------------------------------------------

def test_app_runtime_status_describes_the_strict_session(monkeypatch):
    from app.runtime import runtime_status

    monkeypatch.setattr("ppe.runtime.providers.available_ort_names", lambda: [OV, CPU])
    monkeypatch.setenv("PPE_STRICT_NPU", "1")
    monkeypatch.delenv("PPE_NPU_ONLY", raising=False)
    monkeypatch.delenv("PPE_OPENVINO_DEVICE", raising=False)
    monkeypatch.delenv("PPE_PROVIDERS", raising=False)
    status = runtime_status(prefer_npu=True)
    assert status["strict_npu"] is True and status["npu_only"] is True
    assert status["resolved_providers"] == [OV]  # no CPU EP, even though it is installed
    assert status["openvino_device_type"] == "NPU"  # not the app's usual AUTO:NPU,GPU


def test_app_does_not_default_to_auto_device_under_strict(monkeypatch):
    from app.runtime import _default_openvino_device

    monkeypatch.delenv("PPE_OPENVINO_DEVICE", raising=False)
    monkeypatch.setenv("PPE_STRICT_NPU", "1")
    assert _default_openvino_device(prefer_npu=True) is None  # strict would refuse AUTO and not start
    monkeypatch.delenv("PPE_STRICT_NPU")
    assert _default_openvino_device(prefer_npu=True) == "AUTO:NPU,GPU"  # non-strict unchanged


# --- CLI ----------------------------------------------------------------------------------------

def test_soft_flag_overrides_env_enabled_strict_mode(monkeypatch, tmp_path, capsys):
    pytest.importorskip("onnxruntime")
    from ppe import cli

    monkeypatch.setenv("PPE_STRICT_NPU", "1")
    code = cli.main(["verify-npu", "--soft", "--imgsz", "32", "--weights", str(_tiny_model(tmp_path))])
    out = json.loads(capsys.readouterr().out)
    assert out["strict_npu"] is False  # --soft must win over the environment
    assert out["cpu_resident"] and code == 1  # diagnostic mode lists the ops instead of refusing


def test_strict_flag_absent_keeps_env_strict_for_bench(monkeypatch, tmp_path):
    from ppe import cli

    monkeypatch.setenv("PPE_STRICT_NPU", "1")
    args = cli.build_parser().parse_args(["predict", "--source", "x.jpg"])
    assert cli._policy_from_args(args).strict_npu is True  # not clobbered by an unset flag
    monkeypatch.delenv("PPE_STRICT_NPU")
    assert cli._policy_from_args(args).strict_npu is False


def test_verify_npu_reports_missing_model_as_json_not_traceback(capsys):
    from ppe import cli

    code = cli.main(["verify-npu", "--weights", "definitely/missing.onnx"])
    out = json.loads(capsys.readouterr().out)
    assert code == 1 and out["ok"] is False
    assert out["error_type"] == "FileNotFoundError"


def test_verify_npu_reports_ort_failure_as_json(tmp_path, capsys):
    pytest.importorskip("onnxruntime")
    from ppe import cli

    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not an onnx file")
    code = cli.main(["verify-npu", "--soft", "--weights", str(bad)])
    out = json.loads(capsys.readouterr().out)
    assert code == 1 and out["ok"] is False and out["error_type"]


def test_verify_npu_output_has_no_absolute_repo_path_and_has_fingerprint(tmp_path, capsys):
    pytest.importorskip("onnxruntime")
    from ppe import cli

    model = _tiny_model(tmp_path)
    out_file = tmp_path / "r.json"
    cli.main(["verify-npu", "--soft", "--imgsz", "32", "--weights", str(model), "--out", str(out_file)])
    report = json.loads(out_file.read_text())
    assert report["fingerprint"]["model"]["sha256"]
    assert str(REPO) not in out_file.read_text()


# --- placement cleanup ---------------------------------------------------------------------------

def test_placement_leaves_no_temp_dirs_on_success_or_refusal(tmp_path, monkeypatch):
    pytest.importorskip("onnxruntime")
    from ppe.runtime.placement import verify_placement

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    model = _tiny_model(tmp_path)

    verify_placement(model, ExecutionPolicy(imgsz=32), runs=1)  # success
    monkeypatch.setattr("ppe.runtime.providers.available_ort_names", lambda: [CPU])
    with pytest.raises(RuntimeError, match="NPU execution required"):  # strict refusal
        verify_placement(model, ExecutionPolicy(strict_npu=True), runs=1)
    assert list(scratch.iterdir()) == []


# --- fingerprint caching + scrub -----------------------------------------------------------------

def test_fingerprint_hashes_a_model_once_until_it_changes(tmp_path, monkeypatch):
    from ppe.runtime import fingerprint as fp

    calls = []
    real = fp.file_sha256
    monkeypatch.setattr(fp, "file_sha256", lambda p, **k: (calls.append(p), real(p, **k))[1])
    fp._model_identity.cache_clear()
    m = tmp_path / "m.onnx"
    m.write_bytes(b"one")
    a = fp.runtime_fingerprint(m)["model"]["sha256"]
    fp.runtime_fingerprint(m)
    fp.runtime_fingerprint(m)
    assert len(calls) == 1  # repeated session opens must not re-read the file
    m.write_bytes(b"two-longer")  # size changes -> new cache key
    b = fp.runtime_fingerprint(m)["model"]["sha256"]
    assert len(calls) == 2 and a != b


def test_scrub_paths():
    from ppe.runtime.fingerprint import scrub_paths

    out = scrub_paths({"a": str(REPO / "x" / "y.json"), "b": [str(REPO / "z"), 3], "c": "/other"}, REPO)
    assert out == {"a": "<repo>/x/y.json", "b": ["<repo>/z", 3], "c": "/other"}


# --- strict failure diagnosis -------------------------------------------------------------------

class _Inp:
    def __init__(self, shape):
        self.name, self.shape = "images", shape


def _fake_ort(*, npu_fails=True, cpu_loads=True, cpu_shape=(1, 3, 640, 640)):
    class _SO:
        def add_session_config_entry(self, k, v):
            pass

    class _S:
        def __init__(self, path, providers=None, provider_options=None, sess_options=None):
            if providers == [CPU]:
                if not cpu_loads:
                    raise RuntimeError("INVALID_PROTOBUF")
            elif npu_fails:
                raise RuntimeError("CPU fallback disabled")
            self.providers = providers

        def get_inputs(self):
            return [_Inp(list(cpu_shape))]

        def get_providers(self):
            return self.providers

    return types.SimpleNamespace(SessionOptions=_SO, InferenceSession=_S)


def _strict_open(monkeypatch, tmp_path, ort):
    from ppe.runtime.backends import open_ort_session

    monkeypatch.setattr("ppe.runtime.providers.available_ort_names", lambda: [OV, CPU])
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    return open_ort_session(ort, model, ExecutionPolicy(strict_npu=True, providers=("openvino",)))


def test_strict_failure_names_an_unloadable_model(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="not a loadable ONNX model.*not an NPU problem"):
        _strict_open(monkeypatch, tmp_path, _fake_ort(cpu_loads=False))


def test_strict_failure_names_a_dynamic_shape(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="requires a static input shape"):
        _strict_open(monkeypatch, tmp_path, _fake_ort(cpu_shape=("batch", 3, 640, 640)))


def test_strict_failure_falls_through_to_placement_message(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="loads and has a static shape, but could not be placed"):
        _strict_open(monkeypatch, tmp_path, _fake_ort())


def test_missing_model_file_is_not_reported_as_a_placement_problem(monkeypatch, tmp_path):
    from ppe.runtime.backends import open_ort_session

    monkeypatch.setattr("ppe.runtime.providers.available_ort_names", lambda: [OV, CPU])
    with pytest.raises(FileNotFoundError):
        open_ort_session(_fake_ort(), tmp_path / "nope.onnx", ExecutionPolicy(strict_npu=True))


# --- archived scripts still run from archive/ -----------------------------------------------------

ARCHIVED = ["eval_baseline", "eval_cross_domain", "fetch_checkpoint", "finetune",
            "reorder_checkpoint_classes", "verify_reorder"]


@pytest.mark.parametrize("name", ARCHIVED)
def test_archived_scripts_import_and_show_help(name):
    pytest.importorskip("ultralytics")  # these are training-side tools
    proc = subprocess.run(
        [sys.executable, "-I", str(REPO / "archive" / "scripts" / f"{name}.py"), "--help"],
        capture_output=True, text=True, cwd=REPO, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    assert "usage" in proc.stdout.lower()


def test_no_live_code_points_at_the_old_baselines_or_moved_configs():
    import re

    stale = re.compile(r'REPO_ROOT / "baselines"|root / "baselines"|REPO_ROOT / "configs" / "(finetune|models)"')
    offenders = []
    for base in ("src", "app", "scripts"):
        for py in (REPO / base).rglob("*.py"):
            if stale.search(py.read_text(encoding="utf-8", errors="ignore")):
                offenders.append(str(py.relative_to(REPO)))
    assert offenders == []


# --- smoke export must not clobber tracked weights ------------------------------------------------

def test_smoke_export_does_not_overwrite_the_tracked_onnx(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import smoke_edge
    finally:
        sys.path.remove(str(REPO / "scripts"))

    tracked = tmp_path / "weights"
    tracked.mkdir()
    pt = tracked / "best.pt"
    pt.write_bytes(b"weights")
    (tracked / "best.onnx").write_bytes(b"ORIGINAL-TRACKED")

    class _YOLO:
        def __init__(self, path):
            self.path = Path(path)

        def export(self, **_kw):  # Ultralytics writes next to the .pt it was handed
            target = self.path.with_suffix(".onnx")
            target.write_bytes(b"FRESH-EXPORT")
            return str(target)

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=_YOLO))
    dest = tmp_path / "models" / "best.onnx"
    smoke_edge._ensure_onnx(pt, dest)
    assert dest.read_bytes() == b"FRESH-EXPORT"
    assert (tracked / "best.onnx").read_bytes() == b"ORIGINAL-TRACKED"  # untouched
