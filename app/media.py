"""Image/video helpers for the demo (decode, JPEG encode, capped frame walks)."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

DEFAULT_MAX_VIDEO_FRAMES = 300
DEFAULT_MAX_VIDEO_SECONDS = 45.0
JPEG_QUALITY = 85


def decode_image_bytes(data: bytes) -> np.ndarray:
    if not data:
        raise ValueError("Empty image upload.")
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image. Use JPEG, PNG, or WebP.")
    return image


def encode_jpeg(image: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("Failed to encode annotated JPEG.")
    return buf.tobytes()


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def parse_stream_source(source: str | None) -> int | str:
    """Webcam index (default 0), RTSP URL, or filesystem path."""
    if source is None or not str(source).strip():
        return 0
    text = str(source).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    return text


def open_capture(source: int | str) -> cv2.VideoCapture:
    if isinstance(source, str) and Path(source).is_file():
        return cv2.VideoCapture(str(Path(source)))
    return cv2.VideoCapture(source)


def video_meta(cap: cv2.VideoCapture) -> dict[str, float | int]:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return {"frame_count": total, "fps": fps, "width": width, "height": height}


def frame_stride(frame_count: int, max_frames: int) -> int:
    if max_frames <= 0 or frame_count <= 0 or frame_count <= max_frames:
        return 1
    return max(1, int(math.ceil(frame_count / max_frames)))


def iter_capped_frames(
    cap: cv2.VideoCapture,
    *,
    max_frames: int = DEFAULT_MAX_VIDEO_FRAMES,
    max_seconds: float = DEFAULT_MAX_VIDEO_SECONDS,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (index, BGR frame), sampling across the clip if it is long."""
    meta = video_meta(cap)
    total = int(meta["frame_count"])
    fps = float(meta["fps"])
    stride = frame_stride(total, max_frames)
    max_index = int(max_seconds * fps) if max_seconds > 0 else None
    index = 0
    yielded = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if max_index is not None and index >= max_index:
            break
        if index % stride == 0:
            yield index, frame
            yielded += 1
            if max_frames > 0 and yielded >= max_frames:
                break
        index += 1


def open_video_writer(
    path: Path,
    *,
    fps: float,
    size: tuple[int, int],
) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(float(fps), 1.0),
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer at {path}")
    return writer
