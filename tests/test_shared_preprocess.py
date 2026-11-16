"""Main model + specialists preprocess the frame once, with results identical to the unshared path."""

from __future__ import annotations

import numpy as np
import pytest

from ppe.runtime.backends import Letterboxer, OnnxRuntimeBackend
from ppe.runtime.config import ExecutionPolicy
from ppe.runtime.session import EdgeSession, SpecialistSlot

CPU = "CPUExecutionProvider"


class _Inp:
    def __init__(self, name):
        self.name = name
        self.shape = [1, 3, 64, 64]


class _FakeOrt:
    """One confident detection (class 0, conf 0.9) at the image centre, recording its input tensors."""

    def __init__(self, model_path, providers=None, provider_options=None, sess_options=None):
        self.seen = []
        self._name = "images" if "main" in str(model_path) else "in0"

    def get_inputs(self):
        return [_Inp(self._name)]

    def get_providers(self):
        return [CPU]

    def run(self, _names, feed):
        (tensor,) = feed.values()
        self.seen.append(tensor)
        out = np.zeros((1, 6, 10), dtype=np.float32)
        out[0, :4, 0] = [32, 32, 16, 16]
        out[0, 4, 0] = 0.9
        return [out]


def _backend(monkeypatch, tmp_path, name, names):
    import onnxruntime as ort

    monkeypatch.setattr(ort, "InferenceSession", _FakeOrt)
    monkeypatch.setattr("ppe.runtime.providers.available_ort_names", lambda: [CPU])
    path = tmp_path / name
    path.write_bytes(b"x")
    return OnnxRuntimeBackend(path, ExecutionPolicy(imgsz=64), class_names=names)


def _session(monkeypatch, tmp_path, *, wrap_specialists=lambda b: b):
    main = _backend(monkeypatch, tmp_path, "main.onnx", {0: "person", 1: "helmet"})
    vest = _backend(monkeypatch, tmp_path, "vest.onnx", {0: "vest", 1: "no_vest"})
    helmet = _backend(monkeypatch, tmp_path, "helm.onnx", {0: "helmet", 1: "no_helmet"})
    slots = [
        SpecialistSlot(wrap_specialists(vest), frozenset({"vest", "no_vest"}), tmp_path / "vest.onnx"),
        SpecialistSlot(wrap_specialists(helmet), frozenset({"helmet", "no_helmet"}), tmp_path / "helm.onnx"),
    ]
    sess = EdgeSession(backend=main, policy=main.policy, model_path=tmp_path / "main.onnx",
                       specialist_backends=slots)
    return sess, (main, vest, helmet)


class _NoPrepare:
    """Duck-typed backend without prepare/predict_prepared (like Torch): forces the unshared path."""

    def __init__(self, inner):
        self._inner = inner
        self.name = "noprep"
        self.class_names = inner.class_names

    def predict(self, image_bgr, *, conf):
        return self._inner.predict(image_bgr, conf=conf)

    @property
    def last_timings(self):
        return self._inner.last_timings

    def info(self):
        return {}


FRAME = np.random.default_rng(0).integers(0, 256, size=(48, 64, 3), dtype=np.uint8)


def test_preprocess_runs_once_per_frame_with_two_specialists(monkeypatch, tmp_path):
    sess, _ = _session(monkeypatch, tmp_path)
    calls = []
    orig = Letterboxer.__call__

    def counting(self, image):
        calls.append(1)
        return orig(self, image)

    monkeypatch.setattr(Letterboxer, "__call__", counting)
    sess.predict(FRAME)
    assert len(calls) == 1  # was 3 (main + 2 specialists) before sharing
    sess.predict(FRAME)
    assert len(calls) == 2


def test_all_three_models_receive_the_same_tensor_values(monkeypatch, tmp_path):
    sess, (main, vest, helmet) = _session(monkeypatch, tmp_path)
    sess.predict(FRAME)
    seen = [b.session.seen[0].copy() for b in (main, vest, helmet)]
    assert np.array_equal(seen[0], seen[1]) and np.array_equal(seen[0], seen[2])


def test_shared_path_gives_same_detections_as_unshared_path(monkeypatch, tmp_path):
    shared, _ = _session(monkeypatch, tmp_path)
    unshared, _ = _session(monkeypatch, tmp_path, wrap_specialists=_NoPrepare)
    a = shared.predict(FRAME)
    b = unshared.predict(FRAME)
    key = lambda d: (d.cls_name, round(d.conf, 5), tuple(round(v, 3) for v in d.xyxy))  # noqa: E731
    assert sorted(map(key, a)) == sorted(map(key, b))
    assert {d.cls_name for d in a} >= {"person", "vest"}


def test_timings_do_not_double_count_shared_preprocess(monkeypatch, tmp_path):
    sess, (main, vest, helmet) = _session(monkeypatch, tmp_path)
    sess.predict(FRAME)
    t = sess.last_timings
    assert vest.last_timings["preprocess_ms"] == 0.0 and helmet.last_timings["preprocess_ms"] == 0.0
    assert t["preprocess_ms"] == pytest.approx(main.last_timings["preprocess_ms"])
    assert t["infer_ms"] == pytest.approx(sum(b.last_timings["infer_ms"] for b in (main, vest, helmet)))


def test_falls_back_to_per_backend_preprocess_when_a_backend_cannot_share(monkeypatch, tmp_path):
    sess, _ = _session(monkeypatch, tmp_path, wrap_specialists=_NoPrepare)
    calls = []
    orig = Letterboxer.__call__
    monkeypatch.setattr(Letterboxer, "__call__", lambda self, im: (calls.append(1), orig(self, im))[1])
    sess.predict(FRAME)
    assert len(calls) == 3  # main + each specialist preprocess for themselves


def test_session_without_specialists_does_not_use_the_shared_path(monkeypatch, tmp_path):
    main = _backend(monkeypatch, tmp_path, "main.onnx", {0: "person", 1: "helmet"})
    sess = EdgeSession(backend=main, policy=main.policy, model_path=tmp_path / "main.onnx")
    sess.predict(FRAME)
    assert main.last_timings["preprocess_ms"] > 0.0  # charged normally via predict()
