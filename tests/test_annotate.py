"""Tests for shared BGR annotation helpers."""

from __future__ import annotations

import numpy as np

from ppe.annotate import annotate_frame, draw_compliance
from ppe.compliance import Detection, WorkerCompliance


def test_annotate_frame_draws_without_error():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    dets = [
        Detection("helmet", 0.9, (10, 10, 40, 40)),
        Detection("no_vest", 0.8, (50, 50, 80, 80)),
        Detection("person", 0.95, (5, 5, 100, 110)),
    ]
    workers = [
        WorkerCompliance(
            worker_id=0,
            bbox=(5, 5, 100, 110),
            present=["helmet"],
            missing=["vest"],
            violations=["no_vest", "vest"],
            label="Worker 0 — missing vest",
        )
    ]
    out = annotate_frame(image, dets, workers)
    assert out.shape == image.shape
    assert not np.array_equal(out, image)


def test_draw_compliance_ok_path():
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    workers = [
        WorkerCompliance(
            worker_id=1,
            bbox=(10, 10, 60, 60),
            present=["helmet", "vest"],
            missing=[],
            violations=[],
            label="Worker 1 — compliant",
        )
    ]
    out = draw_compliance(image, workers)
    assert out.shape == image.shape
