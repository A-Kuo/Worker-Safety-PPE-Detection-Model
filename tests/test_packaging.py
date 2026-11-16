"""Wheel-safety and torch-free guarantees (PR 1): paths, packaged tools, dependency split."""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

from ppe import paths
from ppe.runtime.config import resolve_model_path

REPO = Path(__file__).resolve().parents[1]


# --- project root / model dir -------------------------------------------------------------------

def test_source_checkout_is_detected():
    assert paths.source_root() == REPO
    assert paths.project_root() == REPO


def test_project_root_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("PPE_HOME", str(tmp_path))
    assert paths.project_root() == tmp_path


def test_wheel_install_uses_cwd_not_site_packages(monkeypatch, tmp_path):
    # Simulate an installed wheel: this file's grandparent is not a checkout.
    monkeypatch.delenv("PPE_HOME", raising=False)
    monkeypatch.setattr(paths, "source_root", lambda: None)
    monkeypatch.chdir(tmp_path)
    assert paths.project_root() == tmp_path


def test_source_root_is_none_for_an_installed_layout(monkeypatch, tmp_path):
    fake = tmp_path / "site-packages" / "ppe" / "paths.py"
    fake.parent.mkdir(parents=True)
    fake.write_text("")
    monkeypatch.setattr(paths, "__file__", str(fake))
    assert paths.source_root() is None


def test_model_dir_default_and_override(monkeypatch, tmp_path):
    monkeypatch.delenv("PPE_MODEL_DIR", raising=False)
    assert paths.model_dir() == Path.home() / ".ppe" / "models"
    monkeypatch.setenv("PPE_MODEL_DIR", str(tmp_path))
    assert paths.model_dir() == tmp_path


def test_default_model_is_found_in_the_user_model_dir(monkeypatch, tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    (store / "vest_specialist.onnx").write_bytes(b"x")
    monkeypatch.setenv("PPE_MODEL_DIR", str(store))
    empty_project = tmp_path / "proj"
    empty_project.mkdir()
    assert resolve_model_path(None, repo_root=empty_project) == store / "vest_specialist.onnx"


def test_project_local_model_wins_over_the_user_model_dir(monkeypatch, tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    (store / "vest_specialist.onnx").write_bytes(b"x")
    monkeypatch.setenv("PPE_MODEL_DIR", str(store))
    proj = tmp_path / "proj"
    (proj / "models").mkdir(parents=True)
    (proj / "models" / "vest_specialist.onnx").write_bytes(b"y")
    assert resolve_model_path(None, repo_root=proj) == proj / "models" / "vest_specialist.onnx"


def test_missing_model_error_names_where_to_put_it(monkeypatch, tmp_path):
    monkeypatch.setenv("PPE_MODEL_DIR", str(tmp_path / "nowhere"))
    with pytest.raises(FileNotFoundError, match="vest_specialist.onnx"):
        resolve_model_path(None, repo_root=tmp_path)


def test_relative_default_root_follows_cwd_when_installed(monkeypatch, tmp_path):
    monkeypatch.delenv("PPE_HOME", raising=False)
    monkeypatch.setattr(paths, "source_root", lambda: None)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "m.onnx").write_bytes(b"x")
    assert resolve_model_path("m.onnx") == tmp_path / "m.onnx"


def test_predict_without_any_model_is_a_one_line_error(monkeypatch, tmp_path):
    from ppe import cli

    monkeypatch.setenv("PPE_HOME", str(tmp_path))
    monkeypatch.setenv("PPE_MODEL_DIR", str(tmp_path / "none"))
    with pytest.raises(SystemExit, match="ppe predict: No model found"):
        cli.main(["predict", "--source", "frame.jpg"])


# --- torch-free import --------------------------------------------------------------------------

def test_importing_the_runtime_and_cli_never_imports_torch():
    code = (
        "import sys, ppe, ppe.cli, ppe.runtime, ppe.paths, ppe.tools.export, ppe.tools.quantize;"
        "bad=[m for m in ('torch','ultralytics') if m in sys.modules];"
        "assert not bad, bad"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env={**__import__("os").environ, "PYTHONPATH": str(REPO / "src")}, cwd=REPO)
    assert proc.returncode == 0, proc.stderr[-600:]


def test_core_dependencies_are_torch_free_and_runtime_is_an_extra():
    tomllib = pytest.importorskip("tomllib")  # py3.11+
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    core = " ".join(data["project"]["dependencies"]).lower()
    for heavy in ("torch", "ultralytics", "onnxruntime"):
        assert heavy not in core, f"{heavy} must not be a core dependency"
    extras = data["project"]["optional-dependencies"]
    assert any("onnxruntime" in d and "openvino" not in d for d in extras["cpu"])
    assert any("onnxruntime-openvino" in d for d in extras["npu"])
    assert any(d.startswith("openvino") for d in extras["npu"])  # the EP's undeclared runtime DLLs
    assert any("ultralytics" in d for d in extras["train"])


def test_missing_onnxruntime_error_names_both_extras(monkeypatch, tmp_path):
    from ppe.runtime.backends import OnnxRuntimeBackend
    from ppe.runtime.config import ExecutionPolicy

    monkeypatch.setitem(sys.modules, "onnxruntime", None)  # import raises ImportError
    with pytest.raises(ImportError, match=r"ppe\[cpu\].*ppe\[npu\]"):
        OnnxRuntimeBackend(tmp_path / "m.onnx", ExecutionPolicy())


# --- packaged tools -----------------------------------------------------------------------------

def test_cli_does_not_import_from_scripts_dir():
    src = (REPO / "src" / "ppe" / "cli.py").read_text(encoding="utf-8")
    assert "export_onnx" not in src and "quantize_onnx" not in src  # scripts/ is not in the wheel


def test_export_without_ultralytics_says_which_extra_to_install(monkeypatch, tmp_path, capsys):
    from ppe import cli

    monkeypatch.setitem(sys.modules, "ultralytics", None)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"w")
    with pytest.raises(SystemExit, match=r"ppe\[train\]"):
        cli.main(["export", "--weights", str(weights), "--out", str(tmp_path / "o.onnx")])


def test_export_requires_weights_and_reports_missing_file(tmp_path):
    from ppe import cli

    with pytest.raises(SystemExit, match="--weights"):
        cli.main(["export"])
    with pytest.raises(SystemExit, match="Weights not found"):
        cli.main(["export", "--weights", str(tmp_path / "nope.pt")])


def test_export_writes_out_and_never_touches_files_next_to_the_weights(monkeypatch, tmp_path):
    from ppe.tools.export import export_model

    src_dir = tmp_path / "weights"
    src_dir.mkdir()
    pt = src_dir / "best.pt"
    pt.write_bytes(b"w")
    (src_dir / "best.onnx").write_bytes(b"TRACKED")

    class _YOLO:
        def __init__(self, path):
            self.path = Path(path)

        def export(self, **kw):  # Ultralytics writes next to the .pt it was given
            target = self.path.with_suffix(".onnx")
            target.write_bytes(b"FRESH")
            return str(target)

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=_YOLO))
    out = export_model(pt, tmp_path / "models" / "v.onnx")
    assert out.read_bytes() == b"FRESH"
    assert (src_dir / "best.onnx").read_bytes() == b"TRACKED"


def _tiny_model(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper
    import numpy as np

    x = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])
    y = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4, 32, 32])
    w = numpy_helper.from_array(np.random.default_rng(0).normal(size=(4, 3, 1, 1)).astype("float32"), "w")
    node = helper.make_node("Conv", ["images", "w"], ["out"], name="conv0")
    graph = helper.make_graph([node], "g", [x], [y], initializer=[w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "tiny.onnx"
    onnx.save(model, str(path))
    return path


def test_quantize_cli_dynamic_writes_int8_next_to_the_model(tmp_path, capsys):
    pytest.importorskip("onnxruntime")
    from ppe import cli

    model = _tiny_model(tmp_path)
    assert cli.main(["quantize", "--model", str(model)]) == 0
    assert (tmp_path / "tiny.int8.onnx").is_file()


def test_quantize_static_uses_the_runtime_preprocessing(tmp_path):
    pytest.importorskip("onnxruntime")
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from ppe.tools.quantize import quantize_model

    model = _tiny_model(tmp_path)
    calib = tmp_path / "calib"
    calib.mkdir()
    for i in range(3):
        cv2.imwrite(str(calib / f"{i}.png"), np.full((20, 30, 3), 40 * i, dtype=np.uint8))
    (calib / "broken.png").write_bytes(b"not an image")  # unreadable files are skipped, not fatal
    out = quantize_model(model, tmp_path / "q.onnx", calibration=calib, static=True, imgsz=32)
    assert out.is_file() and out.stat().st_size > 0


def test_quantize_errors_are_clean(tmp_path):
    from ppe import cli

    with pytest.raises(SystemExit, match="ONNX model not found"):
        cli.main(["quantize", "--model", str(tmp_path / "nope.onnx")])
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    with pytest.raises(SystemExit, match="calibration"):
        cli.main(["quantize", "--model", str(model), "--static"])
