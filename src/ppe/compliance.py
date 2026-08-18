"""Person–PPE association and per-worker compliance records.

A raw object detector just outputs a bag of boxes ("one helmet, two
persons, one no_vest") with no notion of *whose* helmet or vest it is.
This module answers the actually-useful question: **for each worker, what
gear are they wearing, and what are they missing?**

The core idea is simple: a piece of PPE "belongs" to the person box it sits
inside of (or, failing that, overlaps most with). Once every detection is
assigned to a person, each worker gets a plain-English compliance verdict
like ``"Worker 3 — missing helmet and vest"`` that the API/UI can display
directly with no further interpretation needed.
"""

from __future__ import annotations

from dataclasses import dataclass

from ppe.schema import UNIFIED_CLASS_NAMES

# PPE items we look *for* on a person (the "good" half of each pair).
POSITIVE_PPE = ("helmet", "vest", "goggles", "gloves", "mask")

# Of the positive PPE above, these two are mandatory for a "compliant"
# verdict. Goggles/gloves/mask are tracked and reported but not required —
# adjust this tuple if the safety policy for a given site is stricter.
REQUIRED_ON_PERSON = ("helmet", "vest")

# Explicit "missing X" detections. A no_helmet box is a direct violation
# signal, distinct from simply not detecting a helmet at all (see
# `associate_ppe_to_persons` for how both cases feed into `violations`).
NEGATIVE_VIOLATIONS = ("no_helmet", "no_vest", "no_goggles", "no_gloves", "no_mask")

# Everything that can be "worn" and therefore associated to a person —
# scene classes like `cone`/`ladder`/`fall_detected` are deliberately
# excluded so they never get attached to a worker's compliance record.
_WEARABLE = frozenset(POSITIVE_PPE + NEGATIVE_VIOLATIONS)

# Keeps present/missing/violations lists in the same stable order as
# UNIFIED_CLASS_NAMES, so labels read consistently ("missing helmet and
# vest", never "missing vest and helmet") regardless of detection order.
_NAME_ORDER = {name: i for i, name in enumerate(UNIFIED_CLASS_NAMES)}


@dataclass
class Detection:
    """One raw detector output: a class name, confidence, and box.

    ``cls_name`` is expected to already be in the unified schema (see
    ``ppe.schema.UNIFIED_CLASS_NAMES``) — this module does not remap raw
    dataset labels itself.
    """

    cls_name: str
    conf: float
    xyxy: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixel (or any consistent) units


@dataclass
class WorkerCompliance:
    """The compliance verdict for one detected person.

    ``violations`` is the union of explicit ``no_*`` detections and any
    ``REQUIRED_ON_PERSON`` item that's simply absent — both count as a
    problem worth flagging, even though only the first is a positive
    detection. ``label`` is a ready-to-display string; nothing downstream
    needs to re-derive it from the other fields.
    """

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
    """Associate PPE boxes to person boxes, then score each worker's compliance.

    Association rule, per PPE box, against every person box:
    1. **Containment wins.** If the PPE box's center point falls inside a
       person's box, that's an unambiguous match (a helmet's center is
       basically always over the person wearing it, even in a crowd).
    2. **Otherwise, fall back to IoU.** Useful for PPE that only partially
       overlaps the person box (e.g. a vest box that extends past the
       person's box edge). Matches below ``iou_thresh`` are discarded —
       there's no association, not a weak one.
    3. **Ties broken by person confidence.** A tiny nudge (``1e-3 *
       person.conf``) so that among equally-good geometric matches, the
       more-confident person detection wins.

    Each person then gets exactly one :class:`WorkerCompliance`, in
    detection order (``worker_id`` is the index into the *person* boxes,
    not the original detection list — it is not a stable cross-frame
    identity; use a tracker if you need that).
    """
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
    """Case-insensitive check so both unified (`person`) and raw (`Person`) names work."""
    return name.lower() == "person"


def _best_person(
    ppe: Detection,
    persons: list[Detection],
    iou_thresh: float,
) -> tuple[int | None, float]:
    """Find the best-matching person index for one PPE box (see associate_ppe_to_persons)."""
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
    """True if the center point of ``box`` falls within ``container``."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    cx1, cy1, cx2, cy2 = container
    return cx1 <= cx <= cx2 and cy1 <= cy <= cy2


def _iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Standard Intersection-over-Union for two ``(x1, y1, x2, y2)`` boxes."""
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
    """De-duplicate ``names`` (e.g. two helmet boxes on one person) and sort by schema order."""
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
    """Build the human-readable verdict string shown in the API/UI.

    Missing-required-PPE takes priority in the message over other
    violations (e.g. an explicit no_mask hit) since a bare "no helmet" is
    the higher-severity, more actionable message for a safety reviewer.
    """
    if missing:
        return f"Worker {worker_id} — missing {_join_items(missing)}"
    extras = [v for v in violations if v not in missing]
    if extras:
        return f"Worker {worker_id} — {_join_items(extras)}"
    return f"Worker {worker_id} — compliant"


def _join_items(items: list[str]) -> str:
    """Oxford-comma-free English join: ``"a"``, ``"a and b"``, ``"a, b, and c"``."""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"
