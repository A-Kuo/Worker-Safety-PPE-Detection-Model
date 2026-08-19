"""Greedy IoU tracker that keeps worker identities stable across frames.

Without it every frame renumbers its people, so "worker 3 has been bare-headed
for two seconds" is not a statement the pipeline can make. This is deliberately
simple: no motion model, no appearance features, just box overlap between
consecutive frames. That holds up on fixed site cameras and costs almost
nothing on a low-power device.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

Box = tuple[float, float, float, float]


@dataclass
class Track:
    """One tracked person."""

    track_id: int
    box: Box
    conf: float
    age: int = 0
    hits: int = 1
    misses: int = 0
    history: list[Box] = field(default_factory=list)

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


class IouTracker:
    """Match boxes to existing tracks by overlap, highest overlap first."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 15,
        history_length: int = 30,
    ) -> None:
        if not 0.0 < iou_threshold < 1.0:
            raise ValueError(f"iou_threshold must be in (0, 1), got {iou_threshold}")
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.history_length = int(history_length)
        self._tracks: dict[int, Track] = {}
        self._next_id = 0

    @property
    def tracks(self) -> list[Track]:
        return [self._tracks[key] for key in sorted(self._tracks)]

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 0

    def update(
        self,
        boxes: Sequence[Box],
        confs: Sequence[float] | None = None,
    ) -> list[Track]:
        """Advance one frame and return the track for each input box, in order."""
        scores = list(confs) if confs is not None else [1.0] * len(boxes)
        if len(scores) != len(boxes):
            raise ValueError("boxes and confs must be the same length")

        for track in self._tracks.values():
            track.age += 1

        pairs = self._match(boxes)
        assigned: list[Track] = []

        for index, box in enumerate(boxes):
            track_id = pairs.get(index)
            if track_id is None:
                track = Track(track_id=self._next_id, box=_as_box(box), conf=float(scores[index]))
                self._tracks[track.track_id] = track
                self._next_id += 1
            else:
                track = self._tracks[track_id]
                track.box = _as_box(box)
                track.conf = float(scores[index])
                track.hits += 1
                track.misses = 0
            self._push_history(track)
            assigned.append(track)

        live_ids = {track.track_id for track in assigned}
        for track in list(self._tracks.values()):
            if track.track_id in live_ids:
                continue
            track.misses += 1
            if track.misses > self.max_age:
                del self._tracks[track.track_id]

        return assigned

    def _match(self, boxes: Sequence[Box]) -> dict[int, int]:
        candidates = []
        for det_index, box in enumerate(boxes):
            for track in self._tracks.values():
                score = iou(box, track.box)
                if score >= self.iou_threshold:
                    candidates.append((score, det_index, track.track_id))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

        pairs: dict[int, int] = {}
        used_tracks: set[int] = set()
        for _score, det_index, track_id in candidates:
            if det_index in pairs or track_id in used_tracks:
                continue
            pairs[det_index] = track_id
            used_tracks.add(track_id)
        return pairs

    def _push_history(self, track: Track) -> None:
        track.history.append(track.box)
        if len(track.history) > self.history_length:
            del track.history[0]


def _as_box(box: Sequence[float]) -> Box:
    x1, y1, x2, y2 = (float(v) for v in box)
    return x1, y1, x2, y2


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0
