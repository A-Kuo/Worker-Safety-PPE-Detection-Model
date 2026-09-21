"""Vest specialist: a small 2-class (vest, no_vest) model whose output replaces the main model's.

Why: Combined merges sources that each annotated only some of the 14 classes, so most workers have
no vest status labeled and the 14-class model learns to under-score ``no_vest`` (about 70% of true
``no_vest`` boxes are detected at conf >= 0.01 but only ~24% clear 0.25). Training it on the few
images where vest status IS annotated fixes that (CPU proof of concept: no_vest mAP50 0.22 -> 0.75 on
a fair test), but doing so inside the 14-class model makes it forget every class not labeled in those
images (helmet 0.82 -> 0.00). A separate specialist has nothing else to forget.

At inference the specialist's detections for ``vest`` and ``no_vest`` replace the main model's; every
other class comes from the main model untouched.
"""

from __future__ import annotations

from collections.abc import Iterable

from ppe.compliance import Detection

SPECIALIST_CLASSES = frozenset({"vest", "no_vest"})


def merge_detections(
    main: Iterable[Detection],
    specialist: Iterable[Detection],
    replace: frozenset[str] = SPECIALIST_CLASSES,
) -> list[Detection]:
    """Main detections minus ``replace`` classes, plus the specialist's detections of those classes.

    The specialist may only contribute classes in ``replace``; anything else it emits is ignored.
    """
    kept = [d for d in main if d.cls_name not in replace]
    return kept + [d for d in specialist if d.cls_name in replace]
