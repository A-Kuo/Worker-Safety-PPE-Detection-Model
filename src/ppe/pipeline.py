"""The per-frame path: detect, unify labels, track, score, alert.

:class:`EdgePipeline` is the object a deployment holds on to. It owns the
backend, the tracker, and the alert monitor, and it records how long each stage
takes so a device can report whether it is keeping up with its camera.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from ppe.backends import DetectorBackend, RawDetection, load_backend
from ppe.compliance import Detection, WorkerCompliance, associate_ppe_to_persons, summarize
from ppe.config import RuntimeConfig
from ppe.events import ViolationEvent, ViolationMonitor
from ppe.schema import unify_name
from ppe.tracking import IouTracker

LATENCY_WINDOW = 200


@dataclass
class FrameResult:
    """Everything the pipeline knows about one processed frame."""

    index: int
    timestamp: float
    detections: list[Detection]
    workers: list[WorkerCompliance]
    events: list[ViolationEvent]
    latency_ms: float
    skipped: bool = False

    @property
    def violation_count(self) -> int:
        return sum(len(worker.violations) for worker in self.workers)

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "latency_ms": round(self.latency_ms, 3),
            "detections": [
                {"cls_name": d.cls_name, "conf": round(d.conf, 4), "xyxy": list(d.xyxy)}
                for d in self.detections
            ],
            "compliance": [worker.as_dict() for worker in self.workers],
            "events": [event.describe() for event in self.events],
        }


@dataclass
class LatencyStats:
    """Rolling latency window, summarised the way a device report wants it."""

    samples: deque[float] = field(default_factory=lambda: deque(maxlen=LATENCY_WINDOW))
    total_frames: int = 0

    def add(self, latency_ms: float) -> None:
        self.samples.append(float(latency_ms))
        self.total_frames += 1

    def percentile(self, fraction: float) -> float:
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
        return ordered[index]

    def as_dict(self) -> dict:
        if not self.samples:
            return {"frames": 0}
        mean = sum(self.samples) / len(self.samples)
        return {
            "frames": self.total_frames,
            "window": len(self.samples),
            "mean_ms": round(mean, 2),
            "p50_ms": round(self.percentile(0.50), 2),
            "p95_ms": round(self.percentile(0.95), 2),
            "max_ms": round(max(self.samples), 2),
            "fps": round(1000.0 / mean, 2) if mean > 0 else 0.0,
        }


class EdgePipeline:
    """Detector, tracker, and alerting wired together for streaming use."""

    def __init__(
        self,
        backend: DetectorBackend,
        config: RuntimeConfig | None = None,
        clock=time.monotonic,
    ) -> None:
        self.backend = backend
        self.config = config or RuntimeConfig()
        self.clock = clock
        self.tracker = IouTracker(
            iou_threshold=self.config.track_iou,
            max_age=self.config.track_max_age,
        )
        self.monitor = ViolationMonitor(
            min_frames=self.config.alert_min_frames,
            cooldown_s=self.config.alert_cooldown_s,
        )
        self.latency = LatencyStats()
        self._frame_index = -1

    @classmethod
    def from_config(cls, config: RuntimeConfig) -> EdgePipeline:
        return cls(load_backend(config), config)

    def process(self, frame, timestamp: float | None = None) -> FrameResult:
        """Run one frame end to end."""
        self._frame_index += 1
        index = self._frame_index
        now = self.clock() if timestamp is None else float(timestamp)

        if self.config.frame_stride > 1 and index % self.config.frame_stride:
            return FrameResult(index, now, [], [], [], 0.0, skipped=True)

        started = time.perf_counter()
        detections = [_to_detection(raw) for raw in self.backend.infer(frame)]
        persons = [det for det in detections if det.cls_name.lower() == "person"]
        tracks = self.tracker.update([p.xyxy for p in persons], [p.conf for p in persons])
        workers = associate_ppe_to_persons(
            detections,
            iou_thresh=self.config.associate_iou,
            required=self.config.required_ppe,
            worker_ids=[track.track_id for track in tracks],
        )
        events = self.monitor.update(workers, now)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.latency.add(latency_ms)

        return FrameResult(index, now, detections, workers, events, latency_ms)

    def process_stream(
        self, frames: Iterable, timestamps: Iterable[float] | None = None
    ) -> Iterator[FrameResult]:
        """Process an iterable of frames, skipping strided ones."""
        stamps = iter(timestamps) if timestamps is not None else None
        for frame in frames:
            timestamp = next(stamps, None) if stamps is not None else None
            result = self.process(frame, timestamp)
            if not result.skipped:
                yield result

    def warmup(self, frame, rounds: int | None = None) -> None:
        """Run a few throwaway inferences so the first real frame is not the slowest."""
        for _ in range(self.config.warmup_frames if rounds is None else rounds):
            self.backend.infer(frame)

    def stats(self) -> dict:
        return {
            "backend": getattr(self.backend, "name", type(self.backend).__name__),
            "provider": getattr(self.backend, "provider", None),
            "execution": self.config.execution,
            "frames": self._frame_index + 1,
            "tracks_open": len(self.tracker.tracks),
            "active_violations": len(self.monitor.active()),
            "latency": self.latency.as_dict(),
        }

    def reset(self) -> None:
        self.tracker.reset()
        self.monitor.reset()
        self.latency = LatencyStats()
        self._frame_index = -1

    def close(self) -> None:
        self.backend.close()

    def __enter__(self) -> EdgePipeline:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def summarize_run(results: Iterable[FrameResult]) -> dict:
    """Aggregate a finished run into a report."""
    frames = 0
    workers_max = 0
    violation_frames = 0
    violation_counts: dict[str, int] = {}
    events: list[str] = []

    for result in results:
        frames += 1
        workers_max = max(workers_max, len(result.workers))
        frame_summary = summarize(result.workers)
        if frame_summary["non_compliant"]:
            violation_frames += 1
        for name, count in frame_summary["violation_counts"].items():
            violation_counts[name] = violation_counts.get(name, 0) + count
        events.extend(event.describe() for event in result.events)

    return {
        "frames": frames,
        "workers_max": workers_max,
        "violation_frames": violation_frames,
        "violation_counts": violation_counts,
        "events": events,
    }


def _to_detection(raw: RawDetection) -> Detection:
    return Detection(cls_name=unify_name(raw.cls_name), conf=raw.conf, xyxy=raw.xyxy)
