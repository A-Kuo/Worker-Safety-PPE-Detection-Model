"""Per-class confidence thresholds (lets the weak no_vest class use a lower bar)."""

from __future__ import annotations

import numpy as np
import pytest

from ppe.compliance import Detection
from ppe.inference import PPEDetector
from ppe.runtime.config import ExecutionPolicy
from ppe.runtime.session import EdgeSession
from ppe.thresholds import effective_floor, filter_detections, parse_class_conf


def _det(name: str, conf: float) -> Detection:
    return Detection(cls_name=name, conf=conf, xyxy=(0.0, 0.0, 10.0, 10.0))


def test_parse_class_conf():
    assert parse_class_conf("no_vest=0.05, no_helmet = 0.1") == {"no_vest": 0.05, "no_helmet": 0.1}
    assert parse_class_conf("") == {} and parse_class_conf(None) == {}


@pytest.mark.parametrize("bad", ["no_vest", "no_vest=", "novest=0.1", "no_vest=abc", "no_vest=0", "no_vest=1.5"])
def test_parse_class_conf_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        parse_class_conf(bad)


def test_effective_floor_is_the_lowest_threshold_in_play():
    assert effective_floor(0.25, {}) == 0.25
    assert effective_floor(0.25, {"no_vest": 0.05}) == 0.05
    assert effective_floor(0.25, {"vest": 0.4}) == 0.25  # a higher class bar never raises the floor


def test_filter_applies_each_classs_own_threshold():
    dets = [_det("no_vest", 0.08), _det("no_vest", 0.02), _det("helmet", 0.08), _det("helmet", 0.30)]
    kept = filter_detections(dets, 0.25, {"no_vest": 0.05})
    assert [(d.cls_name, d.conf) for d in kept] == [("no_vest", 0.08), ("helmet", 0.30)]


def test_policy_reads_class_conf_from_env(monkeypatch):
    monkeypatch.setenv("PPE_CLASS_CONF", "no_vest=0.10")
    assert ExecutionPolicy.from_env().class_conf == {"no_vest": 0.10}
    monkeypatch.delenv("PPE_CLASS_CONF")
    assert ExecutionPolicy.from_env().class_conf == {}


class _FakeBackend:
    name = "fake"

    def __init__(self, dets):
        self.dets, self.asked_conf = dets, None

    def predict(self, image_bgr, *, conf):
        self.asked_conf = conf
        return [d for d in self.dets if d.conf >= conf]

    def info(self):
        return {}


def test_edge_session_asks_backend_for_the_floor_then_filters_per_class(tmp_path):
    backend = _FakeBackend([_det("no_vest", 0.08), _det("helmet", 0.08), _det("helmet", 0.5)])
    policy = ExecutionPolicy(conf=0.25, class_conf={"no_vest": 0.05})
    session = EdgeSession(backend=backend, policy=policy, model_path=tmp_path / "m.onnx")
    kept = session.predict(np.zeros((8, 8, 3), np.uint8))
    assert backend.asked_conf == 0.05
    assert sorted((d.cls_name, d.conf) for d in kept) == [("helmet", 0.5), ("no_vest", 0.08)]


class _Boxes:
    def __init__(self, conf, cls):
        self.conf, self.cls = np.asarray(conf, float), np.asarray(cls, float)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, mask):
        return _Boxes(self.conf[mask], self.cls[mask])


class _Result:
    names = {0: "helmet", 1: "no_vest"}

    def __init__(self):
        self.boxes = _Boxes([0.08, 0.08, 0.6], [0, 1, 0])


def test_detector_filters_results_in_place_so_plot_and_detections_agree():
    detector = PPEDetector.__new__(PPEDetector)  # skip loading real weights
    detector.conf, detector.class_conf = 0.25, {"no_vest": 0.05}
    result = _Result()
    detector._apply_class_conf(result)
    assert list(result.boxes.cls) == [1.0, 0.0]  # low-conf helmet dropped, low-conf no_vest and strong helmet kept
    assert list(result.boxes.conf) == [0.08, 0.6]
