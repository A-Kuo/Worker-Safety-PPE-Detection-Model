"""Capture -> inference pipeline for the live window. No GUI code here, so it is testable headless.

Two worker threads and two single-item slots::

    capture thread --> [frame slot] --> inference thread --> [result slot] --> UI

A slot holds only the newest item. If inference is slower than the camera the stale frame is replaced
(and counted as dropped) instead of queueing, so what is on screen is never more than one inference
behind real time. The UI polls the result slot and never blocks on inference.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from ppe.runtime.stats import fps_from_ms, summarize

_EMPTY = object()


class LatestSlot:
    """Thread-safe slot that keeps only the most recent item."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._item: Any = _EMPTY
        self.replaced = 0  # items overwritten before anyone took them (dropped)
        self._closed = False

    def put(self, item: Any) -> None:
        with self._cond:
            if self._item is not _EMPTY:
                self.replaced += 1
            self._item = item
            self._cond.notify_all()

    def put_when_empty(self, item: Any, should_stop: Callable[[], bool]) -> bool:
        """Lossless variant: wait until the previous item was taken. False if stopped first."""
        with self._cond:
            while self._item is not _EMPTY and not self._closed:
                if should_stop():
                    return False
                self._cond.wait(0.05)
            if self._closed:
                return False
            self._item = item
            self._cond.notify_all()
            return True

    def take(self, timeout: float | None = None) -> Any:
        """Remove and return the item, or None on timeout / when closed and empty."""
        with self._cond:
            if self._item is _EMPTY and not self._closed:
                self._cond.wait(timeout)
            if self._item is _EMPTY:
                return None
            item, self._item = self._item, _EMPTY
            self._cond.notify_all()
            return item

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed


class FrameSource(Protocol):
    """Anything that yields BGR frames. ``read`` returns None at end of stream."""

    label: str
    live: bool  # True: camera-like, frames arrive on their own clock (drop when behind)

    def read(self) -> np.ndarray | None: ...

    def close(self) -> None: ...


class CaptureSource:
    """OpenCV camera index, video file or stream URL."""

    def __init__(self, source: int | str, *, loop: bool = False) -> None:
        from ppe.io.capture import open_capture, video_meta

        self._source = source
        self.live = isinstance(source, int) or not Path(str(source)).is_file()
        self._loop = loop and not self.live
        self._cap = open_capture(source, low_latency=self.live)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open video source {source!r}. For a camera check it is connected and not "
                "in use by another app; for a file check the path."
            )
        self.label = f"camera {source}" if isinstance(source, int) else str(source)
        self.fps = float(video_meta(self._cap)["fps"])

    def read(self) -> np.ndarray | None:
        if self.live:
            from ppe.io.capture import read_latest_frame

            ok, frame = read_latest_frame(self._cap, drain=0)
        else:
            ok, frame = self._cap.read()
            if not ok and self._loop:
                import cv2

                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
        return frame if ok and frame is not None else None

    def close(self) -> None:
        self._cap.release()


class ImageSource:
    """A still image repeated: useful to check a model without a camera."""

    live = False
    fps = 30.0

    def __init__(self, path: str | Path, *, frames: int | None = None) -> None:
        import cv2

        self._image = cv2.imread(str(path))
        if self._image is None:
            raise RuntimeError(f"Could not read image {path}")
        self._left = frames
        self.label = str(path)

    def read(self) -> np.ndarray | None:
        if self._left is not None:
            if self._left <= 0:
                return None
            self._left -= 1
        return self._image.copy()

    def close(self) -> None:
        pass


class SyntheticSource:
    """Generated moving-gradient frames: exercises the whole pipeline with no file or camera."""

    live = False
    fps = 30.0
    label = "synthetic"

    def __init__(self, frames: int | None = None, size: tuple[int, int] = (640, 480)) -> None:
        self._left = frames
        self._i = 0
        w, h = size
        self._base = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))

    def read(self) -> np.ndarray | None:
        if self._left is not None:
            if self._left <= 0:
                return None
            self._left -= 1
        self._i += 1
        plane = np.roll(self._base, self._i * 4, axis=1)
        return np.dstack([plane, plane[::-1], 255 - plane])

    def close(self) -> None:
        pass


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def open_source(spec: str | None, *, loop: bool = False, frames: int | None = None) -> FrameSource:
    """``synthetic``, an image path, a video path / URL, or a camera index (default 0)."""
    from ppe.io.capture import parse_stream_source

    if spec is not None and str(spec).strip().lower() == "synthetic":
        return SyntheticSource(frames=frames)
    parsed = parse_stream_source(spec)
    if isinstance(parsed, str) and Path(parsed).suffix.lower() in IMAGE_EXTS:
        return ImageSource(parsed, frames=frames)
    return CaptureSource(parsed, loop=loop)


@dataclass
class FrameResult:
    index: int
    image: np.ndarray  # annotated BGR frame
    counts: dict[str, int]
    timings: dict[str, float]  # stage ms for this frame, incl. total_ms


@dataclass
class PipelineStats:
    """Rolling numbers for the HUD. Read from the UI thread; written by the workers."""

    window: int = 300
    captured: int = 0
    processed: int = 0
    dropped: int = 0
    started: float = field(default_factory=time.perf_counter)
    _totals: deque = field(default_factory=lambda: deque(maxlen=300))
    _stages: dict[str, deque] = field(default_factory=dict)
    _stamps: deque = field(default_factory=lambda: deque(maxlen=120))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, timings: dict[str, float]) -> None:
        with self._lock:
            self.processed += 1
            self._stamps.append(time.perf_counter())
            self._totals.append(timings.get("total_ms", 0.0))
            for key in ("preprocess_ms", "infer_ms", "postprocess_ms"):
                if key in timings:
                    self._stages.setdefault(key, deque(maxlen=self.window)).append(timings[key])

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            totals = list(self._totals)
            stamps = list(self._stamps)
            stages = {k: list(v) for k, v in self._stages.items()}
            captured, processed, dropped = self.captured, self.processed, self.dropped
        lat = summarize(totals)
        fps = 0.0
        if len(stamps) >= 2 and stamps[-1] > stamps[0]:
            fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
        return {
            "captured": captured,
            "processed": processed,
            "dropped": dropped,
            "fps": round(fps, 1),
            "latency_ms": lat,
            "fps_ceiling": fps_from_ms(lat.get("p50", 0.0)),
            "stages_p50_ms": {k: summarize(v).get("p50") for k, v in stages.items()},
        }


class Pipeline:
    """Runs capture and inference on worker threads; the UI takes results from ``results``.

    ``lossless=True`` processes every frame (files, self-test). Otherwise stale frames are dropped,
    which is what a live camera needs.
    """

    def __init__(self, session: Any, source: FrameSource, *, lossless: bool | None = None,
                 pace: bool = True, record: Path | None = None) -> None:
        self.session = session
        self.source = source
        self.lossless = (not source.live) if lossless is None else lossless
        # A file read at full speed would finish in a blink; pace it at its own frame rate unless
        # every frame must be processed as fast as possible (self-test / benchmarking).
        self._interval = (1.0 / getattr(source, "fps", 30.0)) if (pace and not source.live) else 0.0
        self.frames = LatestSlot()
        self.results = LatestSlot()
        self.stats = PipelineStats()
        self.paused = threading.Event()
        self._stop = threading.Event()
        self.finished = threading.Event()  # source ended and everything captured was processed
        self.error: BaseException | None = None
        self._writer = None
        self._writer_lock = threading.Lock()
        self.recording_path: Path | None = None
        # Recording requested up front starts with the very first frame (its size is unknown before).
        self._record_from_start = Path(record) if record else None
        self._threads = [
            threading.Thread(target=self._guard, args=(self._capture_loop,), name="ppe-capture", daemon=True),
            threading.Thread(target=self._guard, args=(self._infer_loop,), name="ppe-infer", daemon=True),
        ]

    # -- lifecycle ---------------------------------------------------------------------------
    def start(self) -> "Pipeline":
        for t in self._threads:
            t.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self.frames.close()
        self.results.close()
        for t in self._threads:
            if t.is_alive() and t is not threading.current_thread():
                t.join(timeout=5)
        self.stop_recording()
        self.source.close()

    def _guard(self, fn: Callable[[], None]) -> None:
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - surface worker failures to the UI thread
            self.error = exc
            self.finished.set()
            self.frames.close()
            self.results.close()

    # -- workers -----------------------------------------------------------------------------
    def _capture_loop(self) -> None:
        index = 0
        next_at = time.perf_counter()
        while not self._stop.is_set():
            if self.paused.is_set():
                time.sleep(0.02)
                next_at = time.perf_counter()
                continue
            frame = self.source.read()
            if frame is None:
                break
            index += 1
            self.stats.captured = index
            if self.lossless:
                if not self.frames.put_when_empty((index, frame), self._stop.is_set):
                    break
            else:
                self.frames.put((index, frame))
                self.stats.dropped = self.frames.replaced
            if self._interval:
                next_at += self._interval
                delay = next_at - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_at = time.perf_counter()
        self.frames.close()  # lets the inference loop drain the last frame and finish

    def _infer_loop(self) -> None:
        from ppe.annotate import annotate_frame

        while not self._stop.is_set():
            item = self.frames.take(timeout=0.1)
            if item is None:
                if self.frames.closed:
                    break
                continue
            index, frame = item
            detections = self.session.predict(frame)
            timings = dict(getattr(self.session, "last_timings", {}) or {})
            counts: dict[str, int] = {}
            for det in detections:
                counts[det.cls_name] = counts.get(det.cls_name, 0) + 1
            image = annotate_frame(frame, detections, None, copy=True)
            self.stats.record(timings)
            if self._record_from_start is not None:
                path, self._record_from_start = self._record_from_start, None
                self.start_recording(path, (image.shape[1], image.shape[0]), fps=getattr(self.source, "fps", 20.0))
            with self._writer_lock:
                if self._writer is not None:
                    self._writer.write(image)
            result = FrameResult(index=index, image=image, counts=counts, timings=timings)
            if self.lossless:
                if not self.results.put_when_empty(result, self._stop.is_set):
                    break
            else:
                self.results.put(result)
        self.finished.set()
        self.results.close()

    # -- controls ----------------------------------------------------------------------------
    def toggle_pause(self) -> bool:
        if self.paused.is_set():
            self.paused.clear()
        else:
            self.paused.set()
        return self.paused.is_set()

    def adjust_conf(self, delta: float) -> float:
        """Change the detection threshold live (clamped to 0.05..0.95)."""
        policy = self.session.policy
        policy.conf = round(min(0.95, max(0.05, policy.conf + delta)), 2)
        return policy.conf

    def start_recording(self, path: Path, size: tuple[int, int], fps: float = 20.0) -> Path:
        from ppe.io.capture import open_video_writer

        with self._writer_lock:
            if self._writer is None:
                self._writer = open_video_writer(Path(path), fps=fps, size=size)
                self.recording_path = Path(path)
        return self.recording_path

    def stop_recording(self) -> Path | None:
        with self._writer_lock:
            path, writer = self.recording_path, self._writer
            self._writer, self.recording_path = None, None
        if writer is not None:
            writer.release()
        return path


def device_label(info: dict[str, Any], placement: dict[str, Any] | None = None) -> str:
    """Short, honest description of where inference runs, for the HUD.

    ``providers_active`` always lists the CPU EP (ORT appends it), so it is never used alone: the
    label is built from the requested provider, the OpenVINO device and, when available, the
    profiled placement.
    """
    backend = info.get("backend", info)
    # Set by the runtime when a requested provider is absent from the live session.
    missing = list(backend.get("providers_missing") or [])
    active = list(backend.get("providers_active") or [])
    primary = next((p for p in active if p != "CPUExecutionProvider"), None) or (active[0] if active else "?")
    name = primary.replace("ExecutionProvider", "")
    device = backend.get("openvino_device_type")
    if name == "OpenVINO" and device:
        name = f"OpenVINO:{device}"
    elif name == "OpenVINO":
        name = "OpenVINO:default device"
    if missing:
        short = ", ".join(m.replace("ExecutionProvider", "") for m in missing)
        name = f"{name} ({short} did not load)"
    strict = " strict" if backend.get("strict_npu") else ""
    if placement:
        if placement.get("ok"):
            frac = placement.get("npu_fraction")
            proof = f" | verified on NPU ({int(round((frac or 0) * 100))}% of nodes)"
        elif placement.get("ok") is False:
            cpu = len(placement.get("cpu_resident") or [])
            proof = f" | NOT NPU-only ({cpu} CPU nodes)" if cpu else " | not pinned to the NPU"
        else:
            proof = " | placement unknown"
    else:
        proof = ""
    return f"{name}{strict}{proof}"


def hud_lines(snapshot: dict[str, Any], counts: dict[str, int], *, device: str, conf: float,
              paused: bool = False, recording: bool = False) -> list[str]:
    lat = snapshot.get("latency_ms", {})
    stages = snapshot.get("stages_p50_ms", {})
    line2 = f"{snapshot.get('fps', 0):.1f} FPS"
    if lat.get("n"):
        line2 += f" | frame p50 {lat['p50']:.1f} ms  p95 {lat['p95']:.1f} ms"
    parts = [f"{label} {stages[key]:.1f}" for key, label in
             (("preprocess_ms", "pre"), ("infer_ms", "infer"), ("postprocess_ms", "post")) if stages.get(key) is not None]
    line3 = ("stages ms: " + "  ".join(parts)) if parts else ""
    vest, no_vest = counts.get("vest", 0), counts.get("no_vest", 0)
    line4 = f"vest {vest}   no_vest {no_vest}   conf {conf:.2f}   dropped {snapshot.get('dropped', 0)}"
    flags = "  ".join(f for f, on in (("PAUSED", paused), ("REC", recording)) if on)
    return [ln for ln in (device, line2, line3, line4, flags) if ln]


def draw_hud(image: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw HUD text on a dark band at the top-left of ``image`` (in place) and return it."""
    import cv2

    if not lines:
        return image
    scale, thick, pad, step = 0.5, 1, 6, 20
    width = max(cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] for ln in lines) + 2 * pad
    height = step * len(lines) + pad
    h, w = image.shape[:2]
    x2, y2 = min(w, width), min(h, height)
    image[0:y2, 0:x2] = (image[0:y2, 0:x2] * 0.35).astype(image.dtype)
    for i, ln in enumerate(lines):
        cv2.putText(image, ln, (pad, step * (i + 1) - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                    thick, cv2.LINE_AA)
    return image
