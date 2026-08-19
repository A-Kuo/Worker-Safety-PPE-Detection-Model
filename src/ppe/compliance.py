"""Association of PPE boxes to people, and the compliance record that follows.

A detector reports helmets and people separately. Everything a site supervisor
actually wants to know ("who is missing a hard hat") lives in the link between
the two, which is what this module builds.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ppe.schema import UNIFIED_CLASS_NAMES

POSITIVE_PPE = ("helmet", "vest", "goggles", "gloves", "mask")
REQUIRED_ON_PERSON = ("helmet", "vest")
NEGATIVE_VIOLATIONS = ("no_helmet", "no_vest", "no_goggles", "no_gloves", "no_mask")

_WEARABLE = frozenset(POSITIVE_PPE + NEGATIVE_VIOLATIONS)
_NAME_ORDER = {name: i for i, name in enumerate(UNIFIED_CLASS_NAMES)}


@dataclass
class Detection:
    cls_name: str
    conf: float
    xyxy: tuple[float, float, float, float]


@dataclass
class WorkerCompliance:
    worker_id: int
    bbox: tuple[float, float, float, float]
    present: list[str]
    missing: list[str]
    violations: list[str]
    label: str

    @property
    def compliant(self) -> bool:
        return not self.violations

    def as_dict(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "bbox": list(self.bbox),
            "present": list(self.present),
            "missing": list(self.missing),
            "violations": list(self.violations),
            "label": self.label,
            "compliant": self.compliant,
        }


def associate_ppe_to_persons(
    detections: Sequence[Detection],
    iou_thresh: float = 0.05,
    required: Sequence[str] = REQUIRED_ON_PERSON,
    worker_ids: Sequence[int] | None = None,
) -> list[WorkerCompliance]:
    """Attach each PPE box to a person and score that person against ``required``.

    A box belongs to the person whose region contains its center; failing that,
    to the person it overlaps most, provided the overlap clears ``iou_thresh``.
    Pass ``worker_ids`` (from the tracker) to keep identities stable between
    frames, otherwise workers are numbered by detection order.
    """
    persons = [det for det in detections if det.cls_name.lower() == "person"]
    wearables = [det for det in detections if det.cls_name in _WEARABLE]

    if worker_ids is not None and len(worker_ids) != len(persons):
        raise ValueError(
            f"worker_ids has {len(worker_ids)} entries for {len(persons)} person boxes"
        )
    ids = list(worker_ids) if worker_ids is not None else list(range(len(persons)))

    assigned: dict[int, list[Detection]] = {i: [] for i in range(len(persons))}
    for ppe in wearables:
        index = _best_person(ppe, persons, iou_thresh)
        if index is not None:
            assigned[index].append(ppe)

    workers: list[WorkerCompliance] = []
    for i, person in enumerate(persons):
        present = _ordered_unique(d.cls_name for d in assigned[i] if d.cls_name in POSITIVE_PPE)
        flagged = _ordered_unique(
            d.cls_name for d in assigned[i] if d.cls_name in NEGATIVE_VIOLATIONS
        )
        missing = [name for name in required if name not in present]
        violations = _ordered_unique([*flagged, *missing])
        workers.append(
            WorkerCompliance(
                worker_id=int(ids[i]),
                bbox=person.xyxy,
                present=present,
                missing=missing,
                violations=violations,
                label=_format_label(int(ids[i]), missing, violations),
            )
        )
    return workers


def summarize(workers: Sequence[WorkerCompliance]) -> dict:
    """Counts suitable for a dashboard tile or a log line."""
    violation_counts: dict[str, int] = {}
    for worker in workers:
        for item in worker.violations:
            violation_counts[item] = violation_counts.get(item, 0) + 1
    compliant = sum(1 for worker in workers if worker.compliant)
    return {
        "workers": len(workers),
        "compliant": compliant,
        "non_compliant": len(workers) - compliant,
        "violation_counts": violation_counts,
    }


def _best_person(ppe: Detection, persons: Sequence[Detection], iou_thresh: float) -> int | None:
    best_index: int | None = None
    best_score = -1.0
    for i, person in enumerate(persons):
        contained = _center_in(ppe.xyxy, person.xyxy)
        overlap = _iou(ppe.xyxy, person.xyxy)
        if not contained and overlap < iou_thresh:
            continue
        # Containment beats overlap; person confidence only breaks ties.
        score = (1.0 if contained else 0.0) + overlap + 1e-3 * person.conf
        if score > best_score:
            best_score = score
            best_index = i
    return best_index


def _center_in(
    box: tuple[float, float, float, float],
    container: tuple[float, float, float, float],
) -> bool:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    cx1, cy1, cx2, cy2 = container
    return cx1 <= cx <= cx2 and cy1 <= cy <= cy2


def _iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _ordered_unique(names: Iterable[str]) -> list[str]:
    out = list(dict.fromkeys(names))
    out.sort(key=lambda n: _NAME_ORDER.get(n, len(_NAME_ORDER)))
    return out


def _format_label(worker_id: int, missing: list[str], violations: list[str]) -> str:
    if missing:
        return f"Worker {worker_id}: missing {_join_items(missing)}"
    extras = [v for v in violations if v not in missing]
    if extras:
        return f"Worker {worker_id}: {_join_items(extras)}"
    return f"Worker {worker_id}: compliant"


def _join_items(items: Sequence[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"
