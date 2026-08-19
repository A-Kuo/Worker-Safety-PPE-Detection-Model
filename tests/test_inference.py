"""The single-image facade."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import HELMET_BOX, PERSON_BOX, VEST_BOX, detection

from ppe.backends import StubBackend
from ppe.config import RuntimeConfig
from ppe.inference import PPEDetector

pytest.importorskip("cv2", reason="annotation needs opencv")

DRESSED = [
    detection("person", PERSON_BOX),
    detection("helmet", HELMET_BOX),
    detection("vest", VEST_BOX),
]


@pytest.fixture
def detector():
    return PPEDetector(
        backend=StubBackend(frames=[DRESSED]),
        config=RuntimeConfig(backend="stub"),
    )


@pytest.fixture
def frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_predict_image_returns_unified_detections(detector, frame):
    names = sorted(det.cls_name for det in detector.predict_image(frame))
    assert names == ["helmet", "person", "vest"]


def test_predict_and_comply_returns_an_annotated_frame(detector, frame):
    annotated, workers = detector.predict_and_comply(frame)
    assert annotated.shape == frame.shape
    assert annotated.max() > 0
    assert workers[0].compliant


def test_each_call_starts_from_a_clean_slate(detector, frame):
    for _ in range(3):
        _annotated, workers = detector.predict_and_comply(frame)
        assert workers[0].worker_id == 0


def test_conf_is_settable(detector):
    detector.conf = 0.6
    assert detector.conf == 0.6
    assert detector._pipeline.config.conf == 0.6


def test_names_come_from_the_backend(detector):
    assert detector.names()[10] == "person"


def test_a_missing_checkpoint_is_reported_clearly():
    with pytest.raises((FileNotFoundError, ImportError, ValueError)):
        PPEDetector("does/not/exist.onnx")
