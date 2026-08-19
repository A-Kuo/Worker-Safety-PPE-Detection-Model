"""Upload decoding and JPEG encoding for the API and the UI.

Capture handling, striding, and mp4 writing come from :mod:`ppe.sources` so the
service and the CLI read video the same way.
"""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np

from ppe.sources import (
    DEFAULT_MAX_FRAMES,
    DEFAULT_MAX_SECONDS,
    capture_info,
    frame_stride,
    iter_capture,
    open_capture,
    open_writer,
    parse_source,
)

DEFAULT_MAX_VIDEO_FRAMES = DEFAULT_MAX_FRAMES
DEFAULT_MAX_VIDEO_SECONDS = DEFAULT_MAX_SECONDS
JPEG_QUALITY = 85

__all__ = [
    "DEFAULT_MAX_VIDEO_FRAMES",
    "DEFAULT_MAX_VIDEO_SECONDS",
    "JPEG_QUALITY",
    "bgr_to_rgb",
    "capture_info",
    "decode_image_bytes",
    "encode_jpeg",
    "frame_stride",
    "iter_capped_frames",
    "open_capture",
    "open_writer",
    "parse_source",
]


def decode_image_bytes(data: bytes) -> np.ndarray:
    if not data:
        raise ValueError("Empty image upload.")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
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


def iter_capped_frames(
    cap: cv2.VideoCapture,
    *,
    max_frames: int = DEFAULT_MAX_VIDEO_FRAMES,
    max_seconds: float = DEFAULT_MAX_VIDEO_SECONDS,
) -> Iterator[tuple[int, np.ndarray, float]]:
    """Yield ``(index, frame, timestamp)`` for a capped, strided walk of a clip."""
    info = capture_info(cap, "upload")
    stride = frame_stride(info.frame_count, max_frames)
    yield from iter_capture(
        cap,
        max_frames=max_frames,
        max_seconds=max_seconds,
        fps=info.fps,
        stride=stride,
    )
