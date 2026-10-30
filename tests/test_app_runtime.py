"""Demo engine prefers ONNX / NPU-oriented policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.paths import resolve_weights_path
from app.runtime import DemoEngine, build_engine, runtime_status
from ppe.compliance import Detection, WorkerCompliance


class _FakeSession:
    def __init__(self):
        self.conf = 0.25
        self.policy = type("P", (), {"conf": 0.25})()
        self.model_path = Path("models/best.onnx")

    def predict(self, image):
        return [Detection("person", 0.9, (1, 1, 10, 10))]

    def predict_and_annotate(self, image):
        workers = [
            WorkerCompliance(0, (1, 1, 10, 10), [], ["helmet", "vest"], ["helmet", "vest"], "Worker 0 — missing helmet and vest")
        ]
        return image.copy(), workers

    def info(self):
        return {
            "model_path": str(self.model_path),
            "backend": {
                "providers_active": ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
                "openvino_device_type": "AUTO:NPU,GPU",
            },
        }


def test_demo_engine_timed(monkeypatch):
    engine = DemoEngine(_FakeSession(), kind="onnxruntime", conf=0.3)
    assert engine.conf == 0.3
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    timed = engine.predict_timed(frame)
    assert timed.latency_ms >= 0
    assert len(timed.workers) == 1
    info = engine.info()
    assert info["engine"] == "onnxruntime"
    assert info["backend"]["openvino_device_type"] == "AUTO:NPU,GPU"


def test_resolve_weights_prefers_onnx(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PPE_WEIGHTS", raising=False)
    models = tmp_path / "models"
    models.mkdir()
    onnx = models / "best.onnx"
    pt = models / "best.pt"
    onnx.write_bytes(b"onnx")
    pt.write_bytes(b"pt")
    # Patch candidates by pointing REPO_ROOT via explicit path is hard; call with explicit None
    # after monkeypatching app.paths.DEFAULT_WEIGHT_CANDIDATES / REPO_ROOT.
    import app.paths as paths

    monkeypatch.setattr(
        paths,
        "DEFAULT_WEIGHT_CANDIDATES",
        (onnx, pt),
    )
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    resolved = resolve_weights_path()
    assert resolved == onnx


def test_runtime_status_reports_prefer_npu(monkeypatch):
    monkeypatch.delenv("PPE_NPU_ONLY", raising=False)
    monkeypatch.delenv("PPE_OPENVINO_DEVICE", raising=False)
    monkeypatch.setattr(
        "ppe.runtime.providers.available_ort_names",
        lambda: ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
    )
    status = runtime_status(prefer_npu=True)
    assert status["prefer_npu"] is True
    assert status["openvino_device_type"] == "AUTO:NPU,GPU"
    assert status["resolved_providers"][0] == "OpenVINOExecutionProvider"


def test_build_engine_uses_onnx_when_present(tmp_path, monkeypatch):
    model = tmp_path / "best.onnx"
    model.write_bytes(b"")

    import onnxruntime as ort

    class _Sess:
        def __init__(self, *a, **k):
            self._providers = k.get("providers") or ["CPUExecutionProvider"]

        def get_inputs(self):
            return [type("I", (), {"name": "images", "shape": [1, 3, 640, 640]})()]

        def get_providers(self):
            return self._providers

        def get_modelmeta(self):
            return type("M", (), {"custom_metadata_map": {}})()

        def run(self, *a, **k):
            return [np.zeros((1, 6, 1), dtype=np.float32)]

    monkeypatch.setattr(ort, "InferenceSession", _Sess)
    monkeypatch.setattr(
        "ppe.runtime.providers.available_ort_names",
        lambda: ["CPUExecutionProvider"],
    )
    engine = build_engine(model, conf=0.2, prefer_npu=True)
    assert engine.kind == "onnxruntime"
    assert isinstance(engine, DemoEngine)


def test_build_engine_missing_weights():
    with pytest.raises(FileNotFoundError):
        build_engine("/no/such/model.onnx")
