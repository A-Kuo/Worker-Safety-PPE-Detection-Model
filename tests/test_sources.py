"""Frame sources, striding, and the errors a bad source produces."""

from __future__ import annotations

import numpy as np
import pytest

from ppe.sources import (
    SourceUnavailable,
    capture_info,
    classify_source,
    frame_stride,
    iter_capture,
    iter_images,
    open_capture,
    open_writer,
    parse_source,
    read_image,
)

cv2 = pytest.importorskip("cv2", reason="frame sources need opencv")


@pytest.fixture
def clip(tmp_path):
    """A 30-frame 64x48 mp4 with a moving bright square."""
    path = tmp_path / "clip.mp4"
    writer = open_writer(path, fps=10.0, size=(64, 48))
    for index in range(30):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        x = 2 + index
        frame[10:30, x : x + 10] = 255
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def still(tmp_path):
    path = tmp_path / "frame.jpg"
    image = np.full((40, 60, 3), 120, dtype=np.uint8)
    cv2.imwrite(str(path), image)
    return path


@pytest.mark.parametrize(
    ("spec", "kind"),
    [
        (None, "camera"),
        ("", "camera"),
        (0, "camera"),
        ("1", "camera"),
        ("rtsp://cam.local/stream", "rtsp"),
        ("https://example.com/live.m3u8", "url"),
        ("/data/site.mp4", "video"),
        ("/data/site.jpg", "image"),
        ("/data/site.PNG", "image"),
    ],
)
def test_classify_source(spec, kind):
    assert classify_source(spec) == kind


@pytest.mark.parametrize(
    ("spec", "expected"),
    [(None, 0), ("", 0), ("2", 2), (3, 3), ("rtsp://x/y", "rtsp://x/y")],
)
def test_parse_source(spec, expected):
    assert parse_source(spec) == expected


def test_open_capture_raises_for_a_missing_file(tmp_path):
    with pytest.raises(SourceUnavailable, match="Could not open source"):
        open_capture(str(tmp_path / "absent.mp4"))


def test_capture_info_reads_clip_metadata(clip):
    cap = open_capture(str(clip))
    try:
        info = capture_info(cap, str(clip))
    finally:
        cap.release()
    assert info.kind == "video"
    assert info.frame_count == 30
    assert info.width == 64
    assert info.height == 48
    assert info.fps == pytest.approx(10.0, abs=0.5)
    assert info.is_live is False


def test_a_live_source_is_marked_live():
    info = capture_info(_FakeCapture(), "rtsp://cam/stream")
    assert info.is_live is True
    assert info.fps == 25.0  # falls back when the source reports nothing


@pytest.mark.parametrize(
    ("count", "cap", "expected"),
    [(0, 10, 1), (10, 100, 1), (100, 10, 10), (95, 10, 10), (101, 10, 11), (50, 0, 1)],
)
def test_frame_stride(count, cap, expected):
    assert frame_stride(count, cap) == expected


def test_iter_capture_walks_every_frame(clip):
    cap = open_capture(str(clip))
    try:
        frames = list(iter_capture(cap, max_frames=0, max_seconds=0, fps=10.0))
    finally:
        cap.release()
    assert len(frames) == 30
    assert frames[0][1].shape == (48, 64, 3)


def test_iter_capture_honours_the_frame_cap(clip):
    cap = open_capture(str(clip))
    try:
        frames = list(iter_capture(cap, max_frames=5, max_seconds=0, fps=10.0))
    finally:
        cap.release()
    assert len(frames) == 5


def test_iter_capture_strides_across_the_clip(clip):
    cap = open_capture(str(clip))
    try:
        frames = list(iter_capture(cap, max_frames=6, max_seconds=0, fps=10.0, stride=5))
    finally:
        cap.release()
    assert [index for index, _frame, _t in frames] == [0, 5, 10, 15, 20, 25]


def test_iter_capture_stops_at_the_time_limit(clip):
    cap = open_capture(str(clip))
    try:
        frames = list(iter_capture(cap, max_frames=0, max_seconds=1.0, fps=10.0))
    finally:
        cap.release()
    assert len(frames) == 10


def test_iter_capture_timestamps_follow_the_frame_rate(clip):
    cap = open_capture(str(clip))
    try:
        stamps = [t for _i, _f, t in iter_capture(cap, max_frames=3, max_seconds=0, fps=10.0)]
    finally:
        cap.release()
    assert stamps == pytest.approx([0.0, 0.1, 0.2])


def test_read_image(still):
    assert read_image(still).shape == (40, 60, 3)


def test_read_image_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_image(tmp_path / "gone.jpg")


def test_read_image_rejects_a_non_image(tmp_path):
    path = tmp_path / "notes.png"
    path.write_text("this is not a png")
    with pytest.raises(ValueError, match="Could not decode"):
        read_image(path)


def test_iter_images_over_a_directory(tmp_path, still):
    second = tmp_path / "b.png"
    cv2.imwrite(str(second), np.zeros((10, 10, 3), dtype=np.uint8))
    (tmp_path / "notes.txt").write_text("ignored")
    names = [path.name for path, _frame in iter_images(tmp_path)]
    assert names == ["b.png", still.name]


def test_iter_images_over_a_single_file(still):
    results = list(iter_images(still))
    assert len(results) == 1
    assert results[0][0] == still


def test_open_writer_creates_parent_directories(tmp_path):
    target = tmp_path / "nested" / "out.mp4"
    writer = open_writer(target, fps=5.0, size=(32, 32))
    writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    assert target.is_file()


class _FakeCapture:
    """Just enough of cv2.VideoCapture for capture_info."""

    def get(self, _prop: int) -> float:
        return 0.0
