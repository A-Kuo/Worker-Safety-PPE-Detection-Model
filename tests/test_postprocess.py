"""Letterboxing, NMS, and YOLO head decoding."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import yolo_tensor

from ppe.postprocess import (
    decode_yolo_output,
    letterbox,
    nms,
    to_input_tensor,
    undo_letterbox,
    xywh_to_xyxy,
)


def test_letterbox_produces_a_square_canvas():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas, info = letterbox(image, 640)
    assert canvas.shape == (640, 640, 3)
    assert info.scale == pytest.approx(1.0)
    assert info.pad_y == pytest.approx(80.0)
    assert info.pad_x == pytest.approx(0.0)


def test_letterbox_preserves_aspect_ratio():
    image = np.zeros((200, 800, 3), dtype=np.uint8)
    _canvas, info = letterbox(image, 640)
    assert info.scale == pytest.approx(0.8)
    assert info.src_width == 800
    assert info.src_height == 200


def test_letterbox_pads_with_grey():
    image = np.full((100, 400, 3), 255, dtype=np.uint8)
    canvas, _info = letterbox(image, 320, fill=114)
    assert canvas[0, 0].tolist() == [114, 114, 114]
    assert canvas[160, 160].tolist() == [255, 255, 255]


def test_letterbox_rejects_a_non_image():
    with pytest.raises(ValueError, match="HxWxC"):
        letterbox(np.zeros((10, 10), dtype=np.uint8), 320)


def test_input_tensor_is_normalised_nchw_rgb():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)  # BGR
    tensor = to_input_tensor(image)
    assert tensor.shape == (1, 3, 64, 64)
    assert tensor.dtype == np.float32
    assert tensor[0, 0, 0, 0] == pytest.approx(30 / 255)  # R channel first
    assert tensor.max() <= 1.0


def test_undo_letterbox_round_trips_a_box():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    _canvas, info = letterbox(image, 640)
    canvas_box = np.array([[100.0, 180.0, 200.0, 280.0]])
    restored = undo_letterbox(canvas_box, info)
    assert restored[0].tolist() == pytest.approx([100.0, 100.0, 200.0, 200.0])


def test_undo_letterbox_clips_to_the_source_frame():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    _canvas, info = letterbox(image, 320)
    restored = undo_letterbox(np.array([[-50.0, -50.0, 900.0, 900.0]]), info)
    assert restored[0].tolist() == pytest.approx([0.0, 0.0, 100.0, 100.0])


def test_xywh_to_xyxy():
    boxes = np.array([[50.0, 50.0, 20.0, 10.0]])
    assert xywh_to_xyxy(boxes)[0].tolist() == [40.0, 45.0, 60.0, 55.0]


def test_nms_keeps_the_highest_scoring_of_an_overlapping_pair():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]])
    scores = np.array([0.6, 0.9])
    assert nms(boxes, scores, 0.5) == [1]


def test_nms_keeps_disjoint_boxes():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]])
    scores = np.array([0.6, 0.9])
    assert sorted(nms(boxes, scores, 0.5)) == [0, 1]


def test_nms_on_an_empty_input():
    assert nms(np.zeros((0, 4)), np.zeros(0), 0.5) == []


def test_decode_drops_boxes_below_the_threshold():
    raw = yolo_tensor([(50, 50, 20, 20, 0, 0.9), (300, 300, 40, 40, 2, 0.10)])
    boxes, scores, class_ids = decode_yolo_output(
        raw, conf_threshold=0.25, iou_threshold=0.45, num_classes=14
    )
    assert boxes.shape == (1, 4)
    assert class_ids.tolist() == [0]
    assert scores[0] == pytest.approx(0.9)


def test_decode_returns_corner_boxes():
    raw = yolo_tensor([(50, 50, 20, 10, 10, 0.8)])
    boxes, _scores, class_ids = decode_yolo_output(raw, 0.25, 0.45, num_classes=14)
    assert boxes[0].tolist() == pytest.approx([40.0, 45.0, 60.0, 55.0])
    assert class_ids[0] == 10


def test_decode_suppresses_duplicates_of_one_class():
    raw = yolo_tensor([(50, 50, 20, 20, 0, 0.9), (51, 51, 20, 20, 0, 0.7)])
    boxes, _scores, _ids = decode_yolo_output(raw, 0.25, 0.45, num_classes=14)
    assert boxes.shape[0] == 1


def test_decode_keeps_overlapping_boxes_of_different_classes():
    raw = yolo_tensor([(50, 50, 20, 20, 0, 0.9), (50, 50, 20, 20, 10, 0.8)])
    _boxes, _scores, class_ids = decode_yolo_output(raw, 0.25, 0.45, num_classes=14)
    assert sorted(class_ids.tolist()) == [0, 10]


def test_decode_accepts_the_transposed_layout():
    raw = yolo_tensor([(50, 50, 20, 20, 3, 0.7)])
    transposed = np.transpose(raw, (0, 2, 1))
    _boxes, _scores, class_ids = decode_yolo_output(transposed, 0.25, 0.45, num_classes=14)
    assert class_ids.tolist() == [3]


def test_decode_honours_max_detections():
    rows = [(10 * i, 10 * i, 5, 5, i % 14, 0.9) for i in range(1, 40)]
    boxes, _scores, _ids = decode_yolo_output(
        yolo_tensor(rows), 0.25, 0.45, max_detections=5, num_classes=14
    )
    assert boxes.shape[0] == 5


def test_decode_with_nothing_above_threshold():
    raw = yolo_tensor([(50, 50, 20, 20, 0, 0.05)])
    boxes, scores, class_ids = decode_yolo_output(raw, 0.25, 0.45, num_classes=14)
    assert boxes.shape == (0, 4)
    assert scores.size == 0
    assert class_ids.size == 0


def test_decode_rejects_a_malformed_tensor():
    with pytest.raises(ValueError, match="at least 5 rows"):
        decode_yolo_output(np.zeros((1, 3, 3), dtype=np.float32), 0.25, 0.45)
