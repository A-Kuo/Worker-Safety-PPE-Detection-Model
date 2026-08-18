"""Unified multi-domain PPE detection library.

Three submodules, each with a single job:

- :mod:`ppe.schema` — the unified 14-class label vocabulary and the
  raw-dataset -> unified remapping tables.
- :mod:`ppe.compliance` — pure-Python logic that turns a bag of detections
  into per-worker "present / missing / violations" verdicts.
- :mod:`ppe.inference` — the Ultralytics YOLO wrapper (:class:`PPEDetector`)
  that ties the two together end-to-end on a real image.

``schema`` and ``compliance`` have zero third-party dependencies and are
fully unit-testable without ultralytics/torch/opencv installed; only
``inference`` needs the full ML stack (and imports it lazily, so importing
``ppe`` itself never requires a GPU).
"""

from ppe.compliance import (
    NEGATIVE_VIOLATIONS,
    REQUIRED_ON_PERSON,
    Detection,
    WorkerCompliance,
    associate_ppe_to_persons,
)
from ppe.inference import PPEDetector
from ppe.schema import (
    COMBINED_RAW_TO_UNIFIED,
    CONSTRUCTION_TO_UNIFIED,
    HHU_TO_UNIFIED,
    SHARED_EVAL_CLASSES,
    UNIFIED_CLASS_NAMES,
    remap_yolo_line,
    unified_id,
    write_dataset_yaml,
)

__all__ = [
    "UNIFIED_CLASS_NAMES",
    "COMBINED_RAW_TO_UNIFIED",
    "CONSTRUCTION_TO_UNIFIED",
    "HHU_TO_UNIFIED",
    "SHARED_EVAL_CLASSES",
    "unified_id",
    "remap_yolo_line",
    "write_dataset_yaml",
    "Detection",
    "WorkerCompliance",
    "REQUIRED_ON_PERSON",
    "NEGATIVE_VIOLATIONS",
    "associate_ppe_to_persons",
    "PPEDetector",
]
