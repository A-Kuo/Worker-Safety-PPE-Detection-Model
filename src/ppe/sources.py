"""Frame sources: still images, video files, webcams, and RTSP URLs.

The CLI, the API, and the Streamlit UI all need the same three things from a
source: open it, walk it without reading a two-hour clip into memory, and know
its nominal frame rate. Keeping that in one place stops the three from drifting
apart.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_MAX_FRAMES = 300
DEFAULT_MAX_SECONDS = 45.0


@dataclass(frozen=True)
class SourceInfo:
    """What a source reports about itself before any frames are read."""

    kind: str
    spec: str
    fps: float = 0.0
    frame_count: int = 0
    width: int = 0
    height: int = 0

    @property
    def is_live(self) -> bool:
        return self.kind in {"camera", "rtsp"}


def classify_source(spec: str | int | None) -> str:
    """Name the kind of source ``spec`` refers to without opening it."""
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        return "camera"
    if isinstance(spec, int) or str(spec).strip().lstrip("-").isdigit():
        return "camera"
    text = str(spec).strip()
    lowered = text.lower()
    if lowered.startswith(("rtsp://", "rtmp://", "http://", "https://")):
        return "rtsp" if lowered.startswith("rtsp") else "url"
    if Path(text).suffix.lower() in IMAGE_SUFFIXES:
        return "image"
    return "video"


def parse_source(spec: str | int | None) -> int | str:
    """Normalise a source spec into what ``cv2.VideoCapture`` expects."""
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        return 0
    if isinstance(spec, int):
        return spec
    text = str(spec).strip()
    return int(text) if text.lstrip("-").isdigit() else text


def open_capture(spec: str | int | None):
    """Open a capture, raising a useful error instead of returning a dead handle."""
    cv2 = _cv2()
    target = parse_source(spec)
    cap = cv2.VideoCapture(target)
    if not cap.isOpened():
        cap.release()
        raise SourceUnavailable(target)
    return cap


def capture_info(cap, spec: str | int | None) -> SourceInfo:
    cv2 = _cv2()
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    return SourceInfo(
        kind=classify_source(spec),
        spec=str(spec if spec is not None else 0),
        fps=fps if fps > 0 else 25.0,
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    )


def frame_stride(frame_count: int, max_frames: int) -> int:
    """Stride that spreads ``max_frames`` samples over the whole clip."""
    if max_frames <= 0 or frame_count <= 0 or frame_count <= max_frames:
        return 1
    return max(1, int(math.ceil(frame_count / max_frames)))


def iter_capture(
    cap,
    *,
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    fps: float = 25.0,
    stride: int | None = None,
) -> Iterator[tuple[int, object, float]]:
    """Yield ``(source_index, frame, timestamp_s)`` under frame and time caps.

    Long clips are sampled across their duration rather than truncated at the
    front, so a summary reflects the whole video.
    """
    step = stride if stride is not None else 1
    limit_index = int(max_seconds * fps) if max_seconds > 0 else None
    index = 0
    yielded = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if limit_index is not None and index >= limit_index:
            break
        if index % step == 0:
            yield index, frame, index / fps if fps > 0 else float(index)
            yielded += 1
            if 0 < max_frames <= yielded:
                break
        index += 1


def read_image(path: str | Path):
    """Load a still image as BGR, raising if the file is missing or unreadable."""
    cv2 = _cv2()
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"No image at {target}")
    image = cv2.imread(str(target), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode {target}; expected JPEG, PNG, or WebP")
    return image


def iter_images(root: str | Path) -> Iterator[tuple[Path, object]]:
    """Walk a file or directory of stills, yielding ``(path, frame)``."""
    target = Path(root)
    paths = (
        [target]
        if target.is_file()
        else sorted(p for p in target.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    )
    for path in paths:
        yield path, read_image(path)


def open_writer(path: str | Path, fps: float, size: tuple[int, int]):
    """Open an mp4 writer, creating the parent directory if needed."""
    cv2 = _cv2()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(target),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(float(fps), 1.0),
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open a video writer at {target}")
    return writer


class SourceUnavailable(RuntimeError):
    """Raised when a camera index, URL, or file cannot be opened."""

    def __init__(self, target: int | str) -> None:
        self.target = target
        super().__init__(
            f"Could not open source {target!r}. Use an integer for a local camera, "
            "an rtsp:// URL for a network camera, or a path to a video file. "
            "Headless hosts have no camera and will always fail on an index."
        )


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "Frame sources need opencv (pip install opencv-python-headless)."
        ) from exc
    return cv2
