"""Backend selection, the stub, and a real ONNX Runtime session.

The ONNX tests build a throwaway graph whose head emits a fixed detection
tensor. That is enough to exercise the parts this repo owns: letterboxing,
decode, NMS, rescaling to the source frame, and reading class names out of
model metadata.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from helpers import PERSON_BOX, detection

from ppe.backends import (
    TORCH_OPT_IN,
    OnnxBackend,
    RawDetection,
    StubBackend,
    UltralyticsBackend,
    _check_input_shape,
    load_backend,
)
from ppe.config import RuntimeConfig
from ppe.providers import NpuUnavailable, available_npu_providers
from ppe.schema import UNIFIED_CLASS_NAMES

onnx = pytest.importorskip("onnx", reason="building a test model needs the onnx package")
pytest.importorskip("onnxruntime", reason="the onnx backend needs onnxruntime")


@pytest.fixture(scope="module")
def onnx_model(tmp_path_factory) -> Path:
    """A graph that ignores its input and emits one person box at 640x640."""
    from onnx import TensorProto, helper, numpy_helper

    num_classes = len(UNIFIED_CLASS_NAMES)
    person_id = UNIFIED_CLASS_NAMES.index("person")
    head = np.zeros((1, 4 + num_classes, 1), dtype=np.float32)
    head[0, :4, 0] = (320.0, 320.0, 160.0, 480.0)  # cxcywh on the letterboxed canvas
    head[0, 4 + person_id, 0] = 0.93

    node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["output0"],
        value=numpy_helper.from_array(head, name="head"),
    )
    graph = helper.make_graph(
        [node],
        "ppe_test",
        inputs=[helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        outputs=[
            helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 4 + num_classes, 1])
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    entry = model.metadata_props.add()
    entry.key = "names"
    entry.value = repr(dict(enumerate(UNIFIED_CLASS_NAMES)))

    path = tmp_path_factory.mktemp("models") / "tiny.onnx"
    onnx.save(model, str(path))
    return path


def test_stub_returns_scripted_frames():
    first = [detection("person", PERSON_BOX)]
    backend = StubBackend(frames=[first, []])
    assert backend.infer(None) == first
    assert backend.infer(None) == []


def test_stub_loops_by_default():
    backend = StubBackend(frames=[[detection("person", PERSON_BOX)]])
    assert backend.infer(None)
    assert backend.infer(None)
    assert backend.calls == 2


def test_stub_can_stop_at_the_end_of_its_script():
    backend = StubBackend(frames=[[detection("person", PERSON_BOX)]], loop=False)
    backend.infer(None)
    assert backend.infer(None) == []


def test_stub_reports_the_unified_vocabulary():
    assert StubBackend().class_names()[10] == "person"


def test_load_backend_picks_the_stub():
    backend = load_backend(RuntimeConfig(backend="stub"))
    assert backend.name == "stub"


def test_load_backend_needs_weights():
    with pytest.raises(ValueError, match="weights path"):
        load_backend(RuntimeConfig(backend="onnx"))


def test_config_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="backend must be one of"):
        RuntimeConfig(backend="tensorrt", weights=Path("x.engine"))


def test_the_torch_backend_needs_an_explicit_opt_in(monkeypatch):
    monkeypatch.delenv(TORCH_OPT_IN, raising=False)
    with pytest.raises(RuntimeError, match="not a deployment target"):
        UltralyticsBackend("best.pt")


def test_the_torch_backend_error_names_the_export_path(monkeypatch):
    monkeypatch.delenv(TORCH_OPT_IN, raising=False)
    with pytest.raises(RuntimeError, match="export_onnx.py"):
        load_backend(RuntimeConfig(backend="ultralytics", weights=Path("best.pt")))


def test_onnx_backend_infers_a_person(onnx_model):
    backend = OnnxBackend(onnx_model, conf=0.25, execution="cpu")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = backend.infer(frame)
    assert len(detections) == 1
    assert isinstance(detections[0], RawDetection)
    assert detections[0].cls_name == "person"
    assert detections[0].conf == pytest.approx(0.93, abs=1e-4)


def test_onnx_boxes_land_in_source_coordinates(onnx_model):
    backend = OnnxBackend(onnx_model, execution="cpu")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    x1, y1, x2, y2 = backend.infer(frame)[0].xyxy
    # 640x480 letterboxes with 80px bars, so the canvas center maps to (320, 240).
    assert (x1 + x2) / 2 == pytest.approx(320.0, abs=1.0)
    assert (y1 + y2) / 2 == pytest.approx(240.0, abs=1.0)
    assert 0 <= x1 < x2 <= 640
    assert 0 <= y1 < y2 <= 480


def test_onnx_respects_the_confidence_threshold(onnx_model):
    backend = OnnxBackend(onnx_model, conf=0.95, execution="cpu")
    assert backend.infer(np.zeros((480, 640, 3), dtype=np.uint8)) == []


def test_onnx_reads_class_names_from_metadata(onnx_model):
    backend = OnnxBackend(onnx_model, execution="cpu")
    assert backend.class_names() == dict(enumerate(UNIFIED_CLASS_NAMES))


def test_onnx_takes_its_input_size_from_the_graph(onnx_model):
    assert OnnxBackend(onnx_model, imgsz=320, execution="cpu").imgsz == 640


def test_onnx_reports_the_provider_it_bound(onnx_model):
    backend = OnnxBackend(onnx_model, execution="cpu")
    assert backend.provider == "CPUExecutionProvider"
    assert backend.on_npu is False
    assert "CPUExecutionProvider" in backend.providers


def test_onnx_missing_file():
    with pytest.raises(FileNotFoundError, match="ONNX model not found"):
        OnnxBackend("nope.onnx", execution="cpu")


def test_strict_npu_refuses_a_host_without_an_npu(onnx_model):
    if available_npu_providers():
        pytest.skip("this host has an NPU provider, so strict mode should succeed")
    with pytest.raises(NpuUnavailable, match="requires an NPU execution provider"):
        OnnxBackend(onnx_model, execution="npu")


def test_npu_preferred_falls_back_to_cpu(onnx_model):
    backend = OnnxBackend(onnx_model, execution="npu-preferred")
    assert backend.provider in {
        "CPUExecutionProvider",
        *(s.name for s in available_npu_providers()),
    }


def test_dynamic_shapes_are_rejected_for_an_npu_provider():
    with pytest.raises(ValueError, match="static input shape"):
        _check_input_shape([1, 3, "height", "width"], "QNNExecutionProvider", "dynamic.onnx")


def test_dynamic_shapes_are_fine_on_cpu():
    _check_input_shape([1, 3, "height", "width"], "CPUExecutionProvider", "dynamic.onnx")


def test_load_backend_builds_the_onnx_path(onnx_model):
    backend = load_backend(RuntimeConfig(weights=onnx_model, execution="cpu"))
    assert backend.name == "onnx"
    assert backend.infer(np.zeros((480, 640, 3), dtype=np.uint8))


def test_auto_backend_never_picks_torch():
    assert RuntimeConfig(weights=Path("best.pt")).backend_name == "onnx"
