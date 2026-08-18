"""Image/video helpers for the demo (decode, JPEG encode, capped frame walks).

Pure OpenCV/NumPy plumbing with no PPE-specific logic — kept separate from
`app/api/main.py` so the endpoint handlers stay focused on request/response
shaping instead of video-IO details. Everything here is safe to unit test
without a real camera or GPU.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

# Long uploaded videos are strided/capped rather than fully processed frame
# by frame — otherwise a multi-minute clip on CPU could take longer than any
# reasonable HTTP request timeout. These are the defaults for
# `POST /predict/video`; both are overridable per-request.
DEFAULT_MAX_VIDEO_FRAMES = 300
DEFAULT_MAX_VIDEO_SECONDS = 45.0
JPEG_QUALITY = 85


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode raw uploaded bytes (JPEG/PNG/WebP) into a BGR numpy array.

    Raises ``ValueError`` — not a decode-library-specific exception — for
    both an empty upload and bytes that don't decode as any supported image
    format, so the API layer can map both to the same HTTP 400 response.
    """
    if not data:
        raise ValueError("Empty image upload.")
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image. Use JPEG, PNG, or WebP.")
    return image


def encode_jpeg(image: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """Encode a BGR numpy image to JPEG bytes (used for API responses and MJPEG frames)."""
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("Failed to encode annotated JPEG.")
    return buf.tobytes()


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert OpenCV's native BGR channel order to RGB (needed before e.g. Streamlit/PIL display)."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def parse_stream_source(source: str | None) -> int | str:
    """Interpret a ``?source=`` query param as a webcam index, URL, or file path.

    A bare integer string (``"0"``, ``"1"``, ...) is treated as a webcam
    device index since that's how OpenCV's ``VideoCapture`` expects it;
    anything else (RTSP URL, file path) is passed through unchanged as a
    string. ``None`` or blank defaults to ``0``, the typical default webcam.
    """
    if source is None or not str(source).strip():
        return 0
    text = str(source).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    return text


def open_capture(source: int | str) -> cv2.VideoCapture:
    """Open a video source, resolving string paths to files explicitly first.

    A plain ``cv2.VideoCapture(source)`` call already handles this, but
    checking ``Path(source).is_file()`` ourselves first avoids OpenCV
    misinterpreting a relative path as something else on some platforms/
    backends, and keeps the intent explicit for readers of this code.
    """
    if isinstance(source, str) and Path(source).is_file():
        return cv2.VideoCapture(str(Path(source)))
    return cv2.VideoCapture(source)


def video_meta(cap: cv2.VideoCapture) -> dict[str, float | int]:
    """Read frame count / fps / dimensions from an opened capture, with safe fallbacks.

    OpenCV can report ``0`` for any of these properties depending on the
    backend/codec (e.g. some webcam drivers don't expose frame count at
    all) — ``fps`` falls back to a reasonable default of 25 rather than 0,
    since a 0 fps would break downstream stride/duration math.
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return {"frame_count": total, "fps": fps, "width": width, "height": height}


def frame_stride(frame_count: int, max_frames: int) -> int:
    """Compute the sampling stride so at most ``max_frames`` frames are read from ``frame_count``.

    Returns ``1`` (read every frame) whenever capping doesn't apply: an
    unknown frame count, a disabled cap (``max_frames <= 0``), or a clip
    already shorter than the cap.
    """
    if max_frames <= 0 or frame_count <= 0 or frame_count <= max_frames:
        return 1
    return max(1, int(math.ceil(frame_count / max_frames)))


def iter_capped_frames(
    cap: cv2.VideoCapture,
    *,
    max_frames: int = DEFAULT_MAX_VIDEO_FRAMES,
    max_seconds: float = DEFAULT_MAX_VIDEO_SECONDS,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(index, BGR frame)`` pairs, sampling evenly across a long clip.

    Two independent caps apply together: ``max_frames`` limits how many
    frames are ever yielded (via :func:`frame_stride`), and ``max_seconds``
    limits how far into the *source* clip's original duration reading goes
    at all (so a strided read of a 2-hour file doesn't still decode the
    entire 2 hours to find its samples). Whichever limit is hit first wins.
    """
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
    """Open an MP4 writer at ``path``, creating parent directories as needed.

    Always uses the widely-compatible ``mp4v`` fourcc. ``fps`` is floored at
    1.0 since OpenCV's writer can behave oddly (or refuse to open) at 0 fps —
    this matters most for strided output, where the *effective* output fps
    (source fps / stride) can otherwise round down to zero on very long clips.
    """
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
