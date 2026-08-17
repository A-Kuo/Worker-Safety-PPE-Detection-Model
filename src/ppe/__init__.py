"""Unified multi-domain PPE detection library."""

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
