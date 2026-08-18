"""PPEDetector tests using mocked YOLO model (no GPU or weights required)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ppe.inference import PPEDetector, _to_unified_name


class _FakeTensor:
    """Minimal tensor-like object for testing _as_numpy."""

    def __init__(self, values):
        self._values = list(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _FakeBoxes:
    """Mimics Ultralytics Boxes with xyxy/conf/cls and __len__."""

    def __init__(self, xyxy, conf, cls, count):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls
        self._count = count

    def __len__(self):
        return self._count


class _FakeResult:
    """Mimics a single Ultralytics prediction result."""

    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes

    def plot(self):
        return "annotated_image"


def _make_fake_result(raw_names: dict[int, str], boxes_data: list[dict]):
    """Build a fake Ultralytics result with .boxes and .names."""
    xyxy = _FakeTensor([b["xyxy"] for b in boxes_data])
    conf = _FakeTensor([b["conf"] for b in boxes_data])
    cls = _FakeTensor([b["cls"] for b in boxes_data])
    boxes = _FakeBoxes(xyxy, conf, cls, len(boxes_data))
    return _FakeResult(raw_names, boxes)


class TestToUnifiedName:
    def test_already_unified(self):
        assert _to_unified_name("helmet") == "helmet"
        assert _to_unified_name("person") == "person"
        assert _to_unified_name("fall_detected") == "fall_detected"

    def test_raw_combined_name(self):
        assert _to_unified_name("Hardhat") == "helmet"
        assert _to_unified_name("NO-Hardhat") == "no_helmet"
        assert _to_unified_name("Safety Vest") == "vest"
        assert _to_unified_name("Fall-Detected") == "fall_detected"

    def test_raw_construction_name(self):
        assert _to_unified_name("Safety Cone") == "cone"

    def test_raw_hhu_name(self):
        assert _to_unified_name("hi-viz helmet") == "helmet"
        assert _to_unified_name("hi-viz vest") == "vest"
        assert _to_unified_name("head") == "no_helmet"

    def test_unknown_name_passes_through(self):
        assert _to_unified_name("unknown_class") == "unknown_class"


class TestPPEDetector:
    @patch("ppe.inference._yolo_cls")
    def test_names_returns_model_names(self, mock_yolo_cls):
        fake_model = MagicMock()
        fake_model.names = {0: "Hardhat", 1: "Person"}
        mock_yolo_cls.return_value = lambda path: fake_model

        detector = PPEDetector("fake_weights.pt")
        assert detector.names() == {0: "Hardhat", 1: "Person"}

    @patch("ppe.inference._yolo_cls")
    def test_predict_image_returns_detections(self, mock_yolo_cls):
        fake_model = MagicMock()
        raw_names = {0: "Hardhat", 1: "Person"}
        result = _make_fake_result(raw_names, [
            {"xyxy": [10.0, 20.0, 50.0, 80.0], "conf": 0.95, "cls": 0},
            {"xyxy": [0.0, 0.0, 100.0, 200.0], "conf": 0.90, "cls": 1},
        ])
        fake_model.predict.return_value = [result]
        mock_yolo_cls.return_value = lambda path: fake_model

        detector = PPEDetector("fake_weights.pt")
        dets = detector.predict_image("test_image.jpg")

        assert len(dets) == 2
        assert dets[0].cls_name == "helmet"
        assert dets[0].conf == 0.95
        assert dets[0].xyxy == (10.0, 20.0, 50.0, 80.0)
        assert dets[1].cls_name == "person"
        assert dets[1].conf == 0.90

    @patch("ppe.inference._yolo_cls")
    def test_predict_image_empty_results(self, mock_yolo_cls):
        fake_model = MagicMock()
        boxes = _FakeBoxes([], [], [], 0)
        result = _FakeResult({}, boxes)
        fake_model.predict.return_value = [result]
        mock_yolo_cls.return_value = lambda path: fake_model

        detector = PPEDetector("fake_weights.pt")
        dets = detector.predict_image("test_image.jpg")
        assert dets == []

    @patch("ppe.inference._yolo_cls")
    def test_predict_image_no_boxes_attr(self, mock_yolo_cls):
        fake_model = MagicMock()
        result = SimpleNamespace(names={})
        fake_model.predict.return_value = [result]
        mock_yolo_cls.return_value = lambda path: fake_model

        detector = PPEDetector("fake_weights.pt")
        dets = detector.predict_image("test_image.jpg")
        assert dets == []

    @patch("ppe.inference._cv2")
    @patch("ppe.inference._yolo_cls")
    def test_predict_and_comply_returns_workers(self, mock_yolo_cls, mock_cv2):
        fake_model = MagicMock()
        raw_names = {0: "Person", 1: "Hardhat", 2: "Safety Vest"}
        result = _make_fake_result(raw_names, [
            {"xyxy": [0.0, 0.0, 100.0, 200.0], "conf": 0.90, "cls": 0},
            {"xyxy": [20.0, 0.0, 60.0, 40.0], "conf": 0.85, "cls": 1},
            {"xyxy": [15.0, 50.0, 85.0, 140.0], "conf": 0.80, "cls": 2},
        ])
        fake_model.predict.return_value = [result]
        mock_yolo_cls.return_value = lambda path: fake_model

        mock_cv2_module = MagicMock()
        mock_cv2.return_value = mock_cv2_module

        detector = PPEDetector("fake_weights.pt")
        annotated, workers = detector.predict_and_comply("test_image.jpg")

        assert len(workers) == 1
        assert workers[0].present == ["helmet", "vest"]
        assert workers[0].missing == []
        assert workers[0].violations == []

    @patch("ppe.inference._yolo_cls")
    def test_predict_conf_and_device_forwarded(self, mock_yolo_cls):
        fake_model = MagicMock()
        result = _FakeResult({}, None)
        fake_model.predict.return_value = [result]
        mock_yolo_cls.return_value = lambda path: fake_model

        detector = PPEDetector("fake_weights.pt", conf=0.5, device="cpu")
        detector.predict_image("test.jpg")

        fake_model.predict.assert_called_once()
        call_kwargs = fake_model.predict.call_args[1]
        assert call_kwargs["conf"] == 0.5
        assert call_kwargs["device"] == "cpu"
        assert call_kwargs["verbose"] is False

    @patch("ppe.inference._yolo_cls")
    def test_predict_no_device(self, mock_yolo_cls):
        fake_model = MagicMock()
        result = _FakeResult({}, None)
        fake_model.predict.return_value = [result]
        mock_yolo_cls.return_value = lambda path: fake_model

        detector = PPEDetector("fake_weights.pt", conf=0.3)
        detector.predict_image("test.jpg")

        call_kwargs = fake_model.predict.call_args[1]
        assert "device" not in call_kwargs
        assert call_kwargs["conf"] == 0.3


class TestDetectorInit:
    def test_missing_ultralytics_raises(self):
        with patch("ppe.inference._yolo_cls", side_effect=ImportError("no ultralytics")):
            with pytest.raises(ImportError, match="no ultralytics"):
                PPEDetector("fake_weights.pt")
