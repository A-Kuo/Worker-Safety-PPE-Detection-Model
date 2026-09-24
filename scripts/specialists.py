"""Registry of class-pair specialists (small models trained only where their classes were annotated).

Combined merges sources that each annotated only some classes, so a class's absence from an image
usually means "not annotated". A specialist for a class pair is trained only on images that annotate
the pair; its classes then replace the main model's (see src/ppe/specialist.py and
docs/experiments.md, "Lifting no_vest"). Adding a specialist = one entry here + one
configs/train/<exp>.yaml + its name in scripts/train.py EXPERIMENTS.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecialistSpec:
    exp: str
    classes: tuple[str, ...]          # unified class names, in the order of the specialist's class ids
    annotated_dir: str                # data/raw/<annotated_dir>: images annotating the classes (make_class_subset)
    external: str | None = None       # extra fully-annotated dataset key (data/processed/<external>), if any


SPECIALISTS: dict[str, SpecialistSpec] = {
    "vest_specialist": SpecialistSpec("vest_specialist", ("vest", "no_vest"), "vest_annotated", "gap_vest"),
    "helmet_specialist": SpecialistSpec("helmet_specialist", ("helmet", "no_helmet"), "helmet_annotated", None),
}


def id_map(spec: SpecialistSpec, unified_names) -> dict[int, int]:
    """Unified class id -> specialist class id."""
    unified = list(unified_names)
    return {unified.index(name): i for i, name in enumerate(spec.classes)}
