"""Image/video helpers for the demo (decode, JPEG encode). Capture helpers live in ``ppe.io.capture``."""

from __future__ import annotations

import cv2
import numpy as np

from app.paths import ensure_src_on_path

ensure_src_on_path()  # `ppe` must be importable when the demo runs from a checkout without install

from ppe.io.capture import (  # noqa: E402,F401 - re-exported for the demo and existing imports
    DEFAULT_MAX_VIDEO_FRAMES,
    DEFAULT_MAX_VIDEO_SECONDS,
    LATEST_FRAME_DRAIN,
    configure_capture_for_latency,
    frame_stride,
    iter_capped_frames,
    open_capture,
    open_video_writer,
    parse_stream_source,
    read_latest_frame,
    video_meta,
)

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
