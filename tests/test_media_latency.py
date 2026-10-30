"""Low-latency camera capture helpers."""

from __future__ import annotations

import numpy as np

from app.media import configure_capture_for_latency, read_latest_frame


class _FakeCap:
    def __init__(self, frames: int):
        self.frames = frames
        self.grabs = 0
        self.props: dict[int, float] = {}

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def grab(self):
        if self.grabs >= self.frames:
            return False
        self.grabs += 1
        return True

    def retrieve(self):
        if self.grabs <= 0:
            return False, None
        return True, np.zeros((8, 8, 3), dtype=np.uint8)


def test_configure_capture_sets_buffersize():
    import cv2

    cap = _FakeCap(1)
    configure_capture_for_latency(cap)
    assert cap.props.get(cv2.CAP_PROP_BUFFERSIZE) == 1


def test_read_latest_frame_drains_buffer():
    cap = _FakeCap(frames=5)
    ok, frame = read_latest_frame(cap, drain=3)
    assert ok is True
    assert frame is not None
    # 1 initial grab + up to 3 drain grabs
    assert cap.grabs == 4


def test_read_latest_frame_empty():
    cap = _FakeCap(frames=0)
    ok, frame = read_latest_frame(cap)
    assert ok is False
    assert frame is None
