"""Person–PPE association and per-worker compliance records."""

from __future__ import annotations

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
    xyxy: tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass
class WorkerCompliance:
    worker_id: int
    bbox: tuple[float, float, float, float]
    present: list[str]
    missing: list[str]
    violations: list[str]  # no_* hits + missing required positives
    label: str  # e.g. "Worker 12 — missing helmet and vest"


def associate_ppe_to_persons(
    detections: list[Detection],
    iou_thresh: float = 0.05,
) -> list[WorkerCompliance]:
    """Associate PPE boxes to person boxes via containment (center-in-person) or IoU."""
    persons = [det for det in detections if _is_person(det.cls_name)]
    wearables = [det for det in detections if det.cls_name in _WEARABLE]

    assigned: dict[int, list[Detection]] = {i: [] for i in range(len(persons))}
    for ppe in wearables:
        best_i, best_score = _best_person(ppe, persons, iou_thresh)
        if best_i is not None:
            assigned[best_i].append(ppe)

    workers: list[WorkerCompliance] = []
    for i, person in enumerate(persons):
        associated = assigned[i]
        present = _ordered_unique(
            det.cls_name for det in associated if det.cls_name in POSITIVE_PPE
        )
        neg_hits = _ordered_unique(
            det.cls_name for det in associated if det.cls_name in NEGATIVE_VIOLATIONS
        )
        missing = [name for name in REQUIRED_ON_PERSON if name not in present]
        violations = _ordered_unique([*neg_hits, *missing])
        workers.append(
            WorkerCompliance(
                worker_id=i,
                bbox=person.xyxy,
                present=present,
                missing=missing,
                violations=violations,
                label=_format_label(i, missing, violations),
            )
        )
    return workers


def _is_person(name: str) -> bool:
    return name.lower() == "person"


def _best_person(
    ppe: Detection,
    persons: list[Detection],
    iou_thresh: float,
) -> tuple[int | None, float]:
    best_i: int | None = None
    best_score = -1.0
    for i, person in enumerate(persons):
        contained = _center_in(ppe.xyxy, person.xyxy)
        iou = _iou(ppe.xyxy, person.xyxy)
        if not contained and iou < iou_thresh:
            continue
        # Prefer containment, then higher IoU, then higher person confidence.
        score = (1.0 if contained else 0.0) + iou + 1e-3 * person.conf
        if score > best_score:
            best_score = score
            best_i = i
    return best_i, best_score


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


def _ordered_unique(names) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    out.sort(key=lambda n: _NAME_ORDER.get(n, len(_NAME_ORDER)))
    return out


def _format_label(worker_id: int, missing: list[str], violations: list[str]) -> str:
    if missing:
        return f"Worker {worker_id} — missing {_join_items(missing)}"
    extras = [v for v in violations if v not in missing]
    if extras:
        return f"Worker {worker_id} — {_join_items(extras)}"
    return f"Worker {worker_id} — compliant"


def _join_items(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"
