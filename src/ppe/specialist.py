"""Class-pair specialists: small 2-class models that replace the main model for those classes.

Why: Combined merges sources that each annotated only some of the 14 classes, so most workers have
no vest (or helmet) status labeled and the 14-class model learns to under-score ``no_*``. Training
it on the few images where the pair IS annotated fixes that pair but makes it forget every class
not labeled there (helmet 0.82 -> 0.00 in a vest fine-tune PoC). A separate specialist has nothing
else to forget.

At inference each specialist's detections for its pair replace the main model's; every other class
comes from the main model untouched. Vest and helmet specialists can both be attached — they own
disjoint class pairs — so the edge / NPU path can lift both product priorities without a second
14-class retrain.
"""

from __future__ import annotations

from collections.abc import Iterable

from ppe.compliance import Detection

# Backward-compatible default: the first shipped specialist was vest/no_vest.
SPECIALIST_CLASSES = frozenset({"vest", "no_vest"})

# Known class-pair specialists (must stay disjoint so multiple can merge sequentially).
KNOWN_SPECIALIST_PAIRS: tuple[frozenset[str], ...] = (
    frozenset({"vest", "no_vest"}),
    frozenset({"helmet", "no_helmet"}),
)


def replace_classes_for(names: Iterable[str]) -> frozenset[str]:
    """Infer which main-model classes a specialist replaces from its emitted / declared names.

    Matches a known pair when any of its names appear; otherwise returns the names as-is (so an
    unknown 2-class specialist still replaces exactly what it owns).
    """
    present = frozenset(n for n in names if n)
    if not present:
        return frozenset()
    for pair in KNOWN_SPECIALIST_PAIRS:
        if present & pair:
            return pair
    return present


def merge_detections(
    main: Iterable[Detection],
    specialist: Iterable[Detection],
    replace: frozenset[str] | None = None,
) -> list[Detection]:
    """Main detections minus ``replace`` classes, plus the specialist's detections of those classes.

    The specialist may only contribute classes in ``replace``; anything else it emits is ignored.
    When ``replace`` is omitted it is inferred from the specialist's class names (falling back to
    vest/no_vest for empty specialist output, matching the historical single-specialist default).
    """
    specialist_list = list(specialist)
    if replace is None:
        replace = replace_classes_for(d.cls_name for d in specialist_list) or SPECIALIST_CLASSES
    kept = [d for d in main if d.cls_name not in replace]
    return kept + [d for d in specialist_list if d.cls_name in replace]
