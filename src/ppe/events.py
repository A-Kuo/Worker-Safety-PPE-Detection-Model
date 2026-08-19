"""Turn per-frame violations into alerts worth acting on.

A detector that fires on every frame produces thousands of identical
notifications a minute, and one that fires on a single frame reports every
flicker of a missed box. The monitor sits between the two: a violation has to
persist for a few consecutive frames before it raises, and it stays quiet for a
cooldown afterwards.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class ViolationEvent:
    """A violation that has persisted long enough to report."""

    track_id: int
    violation: str
    first_seen: float
    raised_at: float
    frames: int
    bbox: Box
    repeat: bool = False

    @property
    def duration_s(self) -> float:
        return max(0.0, self.raised_at - self.first_seen)

    def describe(self) -> str:
        kind = "still" if self.repeat else "now"
        return (
            f"worker {self.track_id} {kind} {self.violation} "
            f"({self.frames} frames, {self.duration_s:.1f}s)"
        )


@dataclass
class _Streak:
    frames: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    last_raised: float | None = None
    bbox: Box = (0.0, 0.0, 0.0, 0.0)


@dataclass
class ViolationMonitor:
    """Debounce per-worker violations into a stream of events."""

    min_frames: int = 3
    cooldown_s: float = 10.0
    forget_after_s: float = 5.0
    _streaks: dict[tuple[int, str], _Streak] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.min_frames < 1:
            raise ValueError(f"min_frames must be >= 1, got {self.min_frames}")
        if self.cooldown_s < 0:
            raise ValueError(f"cooldown_s must be >= 0, got {self.cooldown_s}")

    def update(self, workers: Iterable, timestamp: float) -> list[ViolationEvent]:
        """Fold one frame of compliance records in and return anything raised."""
        events: list[ViolationEvent] = []
        seen: set[tuple[int, str]] = set()

        for worker in workers:
            for violation in worker.violations:
                key = (int(worker.worker_id), str(violation))
                seen.add(key)
                streak = self._streaks.get(key)
                if streak is None:
                    streak = _Streak(first_seen=timestamp)
                    self._streaks[key] = streak
                streak.frames += 1
                streak.last_seen = timestamp
                streak.bbox = tuple(float(v) for v in worker.bbox)  # type: ignore[assignment]
                event = self._maybe_raise(key, streak, timestamp)
                if event is not None:
                    events.append(event)

        self._expire(seen, timestamp)
        return events

    def active(self) -> list[tuple[int, str]]:
        """Keys currently past the debounce threshold."""
        return sorted(
            key for key, streak in self._streaks.items() if streak.frames >= self.min_frames
        )

    def reset(self) -> None:
        self._streaks.clear()

    def _maybe_raise(
        self,
        key: tuple[int, str],
        streak: _Streak,
        timestamp: float,
    ) -> ViolationEvent | None:
        if streak.frames < self.min_frames:
            return None
        if streak.last_raised is not None and timestamp - streak.last_raised < self.cooldown_s:
            return None
        repeat = streak.last_raised is not None
        streak.last_raised = timestamp
        return ViolationEvent(
            track_id=key[0],
            violation=key[1],
            first_seen=streak.first_seen,
            raised_at=timestamp,
            frames=streak.frames,
            bbox=streak.bbox,
            repeat=repeat,
        )

    def _expire(self, seen: set[tuple[int, str]], timestamp: float) -> None:
        for key, streak in list(self._streaks.items()):
            if key in seen:
                continue
            if timestamp - streak.last_seen >= self.forget_after_s:
                del self._streaks[key]
            else:
                streak.frames = 0
