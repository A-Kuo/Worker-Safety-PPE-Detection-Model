"""Unified PPE label taxonomy and per-source class mappings.

See docs/PROJECT_PLAN.md Section 3 and docs/DATASET_NOTES.md ("Unified label
mapping") for the rationale behind each mapping decision. This module is the
single source of truth those docs describe in prose — keep them in sync.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Canonical unified class list. Index in this list == class id written into
# remapped label files. Must match configs/data_unified.yaml exactly.
UNIFIED_CLASSES: list[str] = [
    "helmet",
    "no_helmet",
    "vest",
    "no_vest",
    "goggles",
    "no_goggles",
    "gloves",
    "no_gloves",
    "boots",
    "no_boots",
    "mask",
    "no_mask",
    "person",
    "machinery",
    "vehicle",
    "cone",
    "ladder",
]

UNIFIED_CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(UNIFIED_CLASSES)}

# `None` as a mapping target means "drop boxes of this source class" (documented
# reason required — see DROP_REASONS below). A source class that legitimately has
# no unified equivalent yet (e.g. still pending confirmation from an export we
# don't have access to) should NOT be listed here; leaving it out is what makes
# `remap_source_labels` raise loudly instead of silently discarding data.


@dataclass(frozen=True)
class SourceDataset:
    """One dataset source and its confirmed source-class -> unified-class mapping.

    `status` documents how confirmed this mapping is:
      - "confirmed": we have the real data.yaml/class list and, ideally, real
        images we've inspected.
      - "pending_export": class names are taken from public research (Roboflow
        Universe UI/search snippets), not yet re-verified against an actual
        export — re-check `class_map` against the real `data.yaml` once we have
        API access, per docs/PROJECT_PLAN.md Open Question #3.
    """

    key: str
    display_name: str
    status: str
    class_map: dict[str, str]
    notes: str = ""


DROP_REASONS: dict[tuple[str, str], str] = {
    # (source_key, source_class_name): reason
}

SOURCES: dict[str, SourceDataset] = {
    "construction_site_safety": SourceDataset(
        key="construction_site_safety",
        display_name="Construction Site Safety (snehilsanyal fork, base dataset)",
        status="confirmed",
        class_map={
            "Hardhat": "helmet",
            "Mask": "mask",
            "NO-Hardhat": "no_helmet",
            "NO-Mask": "no_mask",
            "NO-Safety Vest": "no_vest",
            "Person": "person",
            "Safety Cone": "cone",
            "Safety Vest": "vest",
            "machinery": "machinery",
            "vehicle": "vehicle",
        },
        notes=(
            "Confirmed via docs/assets/baseline/data.yaml (real export config from "
            "the upstream repo). No goggles/gloves/boots classes exist in this "
            "source at all - it contributes zero examples for those unified "
            "classes, which worsens their imbalance in the merged dataset."
        ),
    ),
    "ultralytics_construction_ppe": SourceDataset(
        key="ultralytics_construction_ppe",
        display_name="Ultralytics Construction-PPE (official demo dataset)",
        status="confirmed",
        class_map={
            "helmet": "helmet",
            "gloves": "gloves",
            "vest": "vest",
            "boots": "boots",
            "goggles": "goggles",
            # Empirically resolved by drawing this dataset's own boxes on its own
            # images (see docs/DATASET_NOTES.md): despite the literal class name
            # "none", every example places this box over the torso region on a
            # person NOT wearing a hi-vis vest, co-occurring with "no_helmet" over
            # the head region in the same image. It functions as this dataset's
            # no-vest negative, just under a non-descriptive name.
            "none": "no_vest",
            "Person": "person",
            "no_helmet": "no_helmet",
            "no_goggle": "no_goggles",
            "no_gloves": "no_gloves",
            "no_boots": "no_boots",
        },
        notes=(
            "Downloaded directly from "
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip "
            "(no Roboflow API key required, unlike the two datasets you linked). "
            "1,416 images (1132/143/141 train/val/test). This is the *only* source "
            "so far with real, confirmed boots+goggles+gloves data including their "
            "negatives, so it's added here as an interim/supplementary source while "
            "the two Roboflow-hosted datasets remain blocked on an API key."
        ),
    ),
    "ppe_combined_model": SourceDataset(
        key="ppe_combined_model",
        display_name="Personal Protective Equipment - Combined Model (Roboflow Universe)",
        status="pending_export",
        class_map={
            "Hardhat": "helmet",
            "Mask": "mask",
            "Goggles": "goggles",
            "Gloves": "gloves",
            "Safety Vest": "vest",
            "Person": "person",
            "Ladder": "ladder",
            # Confirmed to exist via a third-party browse-query link
            # (NamHoKi/PPE-Detection-for-Construction-Site-Safety README), not
            # yet seen directly in our own export.
            "NO-Safety Vest": "no_vest",
        },
        notes=(
            "PENDING: 44,002 images, version 4 export = 'resize640_allClasses_noAugs' "
            "(70/20/10 split = 30765/8814/4423). Class list above is assembled from "
            "public research, not a real data.yaml - re-verify every key against the "
            "actual export before trusting counts. Known confirmed cross-dataset risk: "
            "this dataset is one of the sources construction_site_safety's own images "
            "were cloned from (see docs/DATASET_NOTES.md) - run dedup against "
            "construction_site_safety specifically once both are available as images."
        ),
    ),
    "hard_hat_universe": SourceDataset(
        key="hard_hat_universe",
        display_name="Hard Hat Universe (Roboflow Universe)",
        status="pending_export",
        class_map={
            "helmet": "helmet",
            "person": "person",
            "hi-viz helmet": "helmet",
            "hi-viz vest": "vest",
            # This dataset's own description states "head" is used explicitly for
            # "an individual may be present without a hard hat" - i.e. it's an
            # explicit (not synthesized) no_helmet equivalent in this source.
            "head": "no_helmet",
            # Generic "hi-viz" (no body-part qualifier) is ambiguous between vest
            # and a broader "wearing hi-vis clothing" concept. Mapped to vest as
            # the closest unified class; flagged for re-review once we have real
            # images to inspect (same technique used to resolve the
            # ultralytics_construction_ppe "none" ambiguity above).
            "hi-viz": "vest",
        },
        notes=(
            "PENDING: version 26 ('no_nulls_plain') = 7,034 images "
            "(4912/1414/708 train/valid/test). Class list confirmed as 6 classes "
            "via Roboflow Universe UI text (head, helmet, person, hi-viz, "
            "hi-viz helmet, hi-viz vest) but not yet re-verified against a real "
            "data.yaml export."
        ),
    ),
}


def remap_source_labels(source_key: str, source_class_names: list[str]) -> list[int | None]:
    """Return, for each index in `source_class_names`, the unified class id to
    remap it to, or None if that source class should be dropped.

    Raises KeyError loudly (rather than silently dropping data) if a source
    dataset's real exported class list contains a name not yet accounted for in
    `SOURCES[source_key].class_map` - this is intentional: an unrecognized class
    name almost always means the export version changed and the mapping table
    (and docs/DATASET_NOTES.md) needs a human decision, not a default.
    """
    source = SOURCES[source_key]
    result: list[int | None] = []
    unmapped: list[str] = []
    for name in source_class_names:
        drop_reason = DROP_REASONS.get((source_key, name))
        if drop_reason is not None:
            result.append(None)
            continue
        unified_name = source.class_map.get(name)
        if unified_name is None:
            unmapped.append(name)
            continue
        result.append(UNIFIED_CLASS_TO_ID[unified_name])
    if unmapped:
        raise KeyError(
            f"Source '{source_key}' has class(es) with no mapping decision yet: "
            f"{unmapped!r}. Add them to SOURCES['{source_key}'].class_map (or "
            f"DROP_REASONS with a documented reason) in src/data/label_schema.py "
            f"before running the merge - do not guess."
        )
    return result
