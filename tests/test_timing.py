"""Per-stage timing, latency statistics and the bench CLI (no NPU hardware needed)."""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest

from ppe.compliance import Detection
from ppe.runtime.config import ExecutionPolicy
from ppe.runtime.session import EdgeSession
from ppe.runtime.stats import fps_from_ms, summarize, summarize_stages


# --- stats -------------------------------------------------------------------------------------

def test_summarize_percentiles_known_sample():
    s = summarize(range(1, 101))  # 1..100
    assert s["n"] == 100
    assert s["min"] == 1 and s["max"] == 100
    assert s["p50"] == pytest.approx(50.5)
    assert s["p95"] == pytest.approx(95.05)
    assert s["p99"] == pytest.approx(99.01)
    assert s["mean"] == pytest.approx(50.5)


def test_summarize_empty_and_single():
    assert summarize([]) == {"n": 0}
    one = summarize([7.5])
    assert one["n"] == 1 and one["p50"] == one["p99"] == one["max"] == 7.5


def test_fps_from_ms():
    assert fps_from_ms(20.0) == 50.0
    assert fps_from_ms(0.0) == 0.0


def test_summarize_stages_orders_known_stages_first_and_skips_missing():
    samples = [
        {"infer_ms": 10.0, "preprocess_ms": 1.0, "total_ms": 12.0},
        {"infer_ms": 14.0, "preprocess_ms": 3.0, "total_ms": 18.0},
        {"total_ms": 20.0},  # a backend that cannot split stages
    ]
    out = summarize_stages(samples)
    assert list(out) == ["preprocess_ms", "infer_ms", "total_ms"]
    assert out["infer_ms"]["n"] == 2 and out["infer_ms"]["mean"] == 12.0
    assert out["total_ms"]["n"] == 3


# --- backend + session timings ------------------------------------------------------------------

class _Inp:
    name = "images"
    shape = [1, 3, 64, 64]


class _FakeOrtSession:
    """Returns one confident detection in raw YOLO layout (1, 4+nc, N)."""

    def __init__(self, model_path, providers=None, provider_options=None, sess_options=None):
        self.providers = providers

    def get_inputs(self):
        return [_Inp()]

    def get_providers(self):
        return self.providers

    def run(self, _names, feed):
        assert feed["images"].shape == (1, 3, 64, 64)
        out = np.zeros((1, 6, 10), dtype=np.float32)  # 4 box coords + 2 classes, 10 anchors
        out[0, :4, 0] = [32, 32, 16, 16]
        out[0, 4, 0] = 0.9
        return [out]


def _backend(monkeypatch, tmp_path):
    import onnxruntime as ort

    from ppe.runtime.backends import OnnxRuntimeBackend

    monkeypatch.setattr(ort, "InferenceSession", _FakeOrtSession)
    monkeypatch.setattr("ppe.runtime.providers.available_ort_names", lambda: ["CPUExecutionProvider"])
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    return OnnxRuntimeBackend(model, ExecutionPolicy(imgsz=64), class_names={0: "helmet", 1: "vest"})


def test_backend_records_stage_timings(monkeypatch, tmp_path):
    be = _backend(monkeypatch, tmp_path)
    assert be.last_timings == {}  # nothing before the first predict
    dets = be.predict(np.zeros((48, 64, 3), dtype=np.uint8), conf=0.25)
    assert [d.cls_name for d in dets] == ["helmet"]
    t = be.last_timings
    assert set(t) == {"preprocess_ms", "infer_ms", "postprocess_ms"}
    assert all(v >= 0 for v in t.values())


def test_backend_timings_are_thread_local(monkeypatch, tmp_path):
    be = _backend(monkeypatch, tmp_path)
    be.predict(np.zeros((48, 64, 3), dtype=np.uint8), conf=0.25)
    seen = {}

    def other():
        seen["before"] = be.last_timings  # a new thread must not see this thread's numbers
        be.predict(np.zeros((48, 64, 3), dtype=np.uint8), conf=0.25)
        seen["after"] = be.last_timings

    th = threading.Thread(target=other)
    th.start()
    th.join()
    assert seen["before"] == {}
    assert set(seen["after"]) == {"preprocess_ms", "infer_ms", "postprocess_ms"}
    assert set(be.last_timings) == {"preprocess_ms", "infer_ms", "postprocess_ms"}


class _StubBackend:
    name = "stub"

    def __init__(self, dets, timings):
        self._dets, self._timings = dets, timings

    @property
    def last_timings(self):
        return dict(self._timings)

    def predict(self, image_bgr, *, conf):
        return list(self._dets)

    def info(self):
        return {}


def test_session_sums_stage_timings_across_main_and_specialist(tmp_path):
    from ppe.runtime.session import SpecialistSlot

    main = _StubBackend([Detection("person", 0.9, (0, 0, 10, 10))],
                        {"preprocess_ms": 1.0, "infer_ms": 10.0, "postprocess_ms": 0.5})
    spec = _StubBackend([Detection("vest", 0.8, (0, 0, 5, 5))],
                        {"preprocess_ms": 2.0, "infer_ms": 4.0, "postprocess_ms": 0.25})
    sess = EdgeSession(
        backend=main,
        policy=ExecutionPolicy(),
        model_path=tmp_path / "m.onnx",
        specialist_backends=[SpecialistSlot(backend=spec, replace=frozenset({"vest", "no_vest"}),
                                            path=tmp_path / "s.onnx")],
    )
    assert sess.last_timings == {}
    sess.predict(np.zeros((8, 8, 3), dtype=np.uint8))
    t = sess.last_timings
    assert t["preprocess_ms"] == pytest.approx(3.0)
    assert t["infer_ms"] == pytest.approx(14.0)
    assert t["postprocess_ms"] == pytest.approx(0.75)
    assert t["merge_ms"] >= 0
    assert t["total_ms"] >= t["merge_ms"]


def test_session_without_stage_data_still_reports_total(tmp_path):
    sess = EdgeSession(backend=_StubBackend([], {}), policy=ExecutionPolicy(),
                       model_path=tmp_path / "m.onnx")
    sess.predict(np.zeros((8, 8, 3), dtype=np.uint8))
    assert set(sess.last_timings) == {"merge_ms", "total_ms"}


def test_scrub_paths_removes_repo_root_everywhere_and_leaves_other_values():
    from pathlib import Path

    from ppe.runtime.fingerprint import scrub_paths

    root = Path.cwd()
    data = {
        "model": str(root / "models" / "best.onnx"),
        "nested": {"p": [str(root / "a" / "b.json"), 3, None, "relative/x"]},
        "elsewhere": "/opt/other/place",
    }
    out = scrub_paths(data, root)
    assert out["model"] == "<repo>/models/best.onnx"
    assert out["nested"]["p"] == ["<repo>/a/b.json", 3, None, "relative/x"]
    assert out["elsewhere"] == "/opt/other/place"
    assert str(root) not in json.dumps(out) and str(root).replace("\\", "\\\\") not in json.dumps(out)


# --- bench CLI (real ORT, tiny generated model) ------------------------------------------------

def _tiny_model(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])
    y = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 6, 8])
    # Reshape a (1,3,32,32) tensor to (1,6,8)-ish via flatten+slice is overkill; use a fixed
    # constant output so decode has valid layout without any detections above threshold.
    const = helper.make_tensor("c", TensorProto.FLOAT, [1, 6, 8], [0.0] * 48)
    node = helper.make_node("Constant", [], ["out"], value=const, name="const0")
    graph = helper.make_graph([node], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "tiny.onnx"
    onnx.save(model, str(path))
    return path


def test_bench_reports_percentiles_stages_and_fingerprint(tmp_path, capsys):
    pytest.importorskip("onnxruntime")
    from ppe import cli

    out = tmp_path / "bench.json"
    code = cli.main([
        "bench", "--weights", str(_tiny_model(tmp_path)), "--imgsz", "32",
        "--warmup", "2", "--iterations", "12", "--tag", "unit", "--out", str(out),
    ])
    assert code == 0
    result = json.loads(out.read_text())
    assert result["iterations"] == 12 and result["tag"] == "unit"
    assert result["latency_ms"]["n"] == 12
    for key in ("p50", "p95", "p99", "max", "mean"):
        assert key in result["latency_ms"]
    assert list(result["stages_ms"])[:3] == ["preprocess_ms", "infer_ms", "postprocess_ms"]
    assert result["stages_ms"]["infer_ms"]["n"] == 12
    assert result["fingerprint"]["model"]["sha256"]
    assert result["note"].startswith("fewer than 100")  # honesty marker for short runs
    assert json.loads(capsys.readouterr().out.split("Wrote")[0])["iterations"] == 12  # single JSON doc
