"""PPE detection for site cameras: unified labels, compliance, edge runtime."""

from ppe.compliance import (
    NEGATIVE_VIOLATIONS,
    POSITIVE_PPE,
    REQUIRED_ON_PERSON,
    Detection,
    WorkerCompliance,
    associate_ppe_to_persons,
    summarize,
)
from ppe.config import RuntimeConfig, config_from_env
from ppe.events import ViolationEvent, ViolationMonitor
from ppe.pipeline import EdgePipeline, FrameResult, LatencyStats, summarize_run
from ppe.schema import (
    COMBINED_RAW_TO_UNIFIED,
    CONSTRUCTION_TO_UNIFIED,
    HHU_TO_UNIFIED,
    RAW_TO_UNIFIED,
    SHARED_EVAL_CLASSES,
    UNIFIED_CLASS_NAMES,
    remap_yolo_line,
    unified_id,
    unify_name,
    write_dataset_yaml,
)
from ppe.tracking import IouTracker, Track

__version__ = "0.2.0"

__all__ = [
    "COMBINED_RAW_TO_UNIFIED",
    "CONSTRUCTION_TO_UNIFIED",
    "HHU_TO_UNIFIED",
    "NEGATIVE_VIOLATIONS",
    "POSITIVE_PPE",
    "RAW_TO_UNIFIED",
    "REQUIRED_ON_PERSON",
    "SHARED_EVAL_CLASSES",
    "UNIFIED_CLASS_NAMES",
    "Detection",
    "EdgePipeline",
    "FrameResult",
    "IouTracker",
    "LatencyStats",
    "PPEDetector",
    "RuntimeConfig",
    "Track",
    "ViolationEvent",
    "ViolationMonitor",
    "WorkerCompliance",
    "__version__",
    "associate_ppe_to_persons",
    "config_from_env",
    "remap_yolo_line",
    "summarize",
    "summarize_run",
    "unified_id",
    "unify_name",
    "write_dataset_yaml",
]


def __getattr__(name: str):
    # PPEDetector pulls in a backend, which may pull in torch. Keep it off the
    # import path until something actually asks for it.
    if name == "PPEDetector":
        from ppe.inference import PPEDetector

        return PPEDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
