"""Reused-buffer letterbox must be bit-identical to the original allocate-per-frame version."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from ppe.runtime.backends import Letterboxer, _letterbox


def _reference_letterbox(image_bgr: np.ndarray, imgsz: int):
    """The original implementation (pre-optimization), kept verbatim as the oracle."""
    import cv2

    h0, w0 = image_bgr.shape[:2]
    scale = min(imgsz / h0, imgsz / w0)
    nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
    resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    top = (imgsz - nh) // 2
    left = (imgsz - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
    meta = {"scale": scale, "pad": (left, top), "orig": (w0, h0), "imgsz": imgsz}
    return tensor, meta


def _img(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize(
    "h,w",
    [(683, 1024), (1024, 683), (640, 640), (480, 640), (1080, 1920), (33, 47), (1, 64), (700, 701)],
)
def test_letterboxer_is_bit_identical_to_reference(h, w):
    img = _img(h, w)
    ref_t, ref_m = _reference_letterbox(img, 640)
    t, m = Letterboxer(640)(img)
    assert m == ref_m
    assert t.shape == (1, 3, 640, 640) and t.dtype == np.float32
    assert np.array_equal(t, ref_t)  # exact, not allclose


def test_output_is_contiguous_unlike_the_old_transpose_view():
    t, _ = Letterboxer(64)(_img(48, 64))
    assert t.flags["C_CONTIGUOUS"]
    ref_t, _ = _reference_letterbox(_img(48, 64), 64)
    assert not ref_t.flags["C_CONTIGUOUS"]  # documents why ORT used to copy it again


def test_reuses_buffers_across_calls_and_clears_stale_padding():
    lb = Letterboxer(64)
    wide, _ = lb(_img(32, 64, seed=1))
    assert wide is lb._tensor
    first = wide.copy()
    tall, m = lb(_img(64, 32, seed=2))  # different pad region: stale pixels must not leak through
    assert tall is wide  # same buffer object, by design
    ref_t, _ = _reference_letterbox(_img(64, 32, seed=2), 64)
    assert np.array_equal(tall, ref_t)
    assert not np.array_equal(first, tall)


def test_stateless_helper_returns_independent_arrays():
    a, _ = _letterbox(_img(48, 64, seed=3), 64)
    b, _ = _letterbox(_img(48, 64, seed=4), 64)
    assert a is not b and not np.array_equal(a, b)


def test_backend_gives_each_thread_its_own_buffers(monkeypatch, tmp_path):
    import onnxruntime as ort

    from ppe.runtime.backends import OnnxRuntimeBackend
    from ppe.runtime.config import ExecutionPolicy

    class _Inp:
        name = "images"
        shape = [1, 3, 64, 64]

    class _S:
        def __init__(self, *a, **k):
            pass

        def get_inputs(self):
            return [_Inp()]

        def get_providers(self):
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(ort, "InferenceSession", _S)
    monkeypatch.setattr("ppe.runtime.providers.available_ort_names", lambda: ["CPUExecutionProvider"])
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    be = OnnxRuntimeBackend(model, ExecutionPolicy(imgsz=64), class_names={0: "a"})

    main_lb = be._letterboxer()
    assert be._letterboxer() is main_lb  # stable within a thread
    seen = {}
    th = threading.Thread(target=lambda: seen.update(lb=be._letterboxer()))
    th.start()
    th.join()
    assert seen["lb"] is not main_lb and seen["lb"]._tensor is not main_lb._tensor

    be.policy.imgsz = 96  # a changed imgsz must rebuild, never reuse a wrong-sized buffer
    assert be._letterboxer().imgsz == 96
