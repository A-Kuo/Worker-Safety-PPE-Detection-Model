"""Annotation output."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import HELMET_BOX, PERSON_BOX

from ppe.compliance import Detection, associate_ppe_to_persons
from ppe.draw import annotate, draw_banner

pytest.importorskip("cv2", reason="annotation needs opencv")


@pytest.fixture
def blank():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_annotate_leaves_the_input_untouched(blank):
    detections = [Detection("helmet", 0.9, HELMET_BOX)]
    annotate(blank, detections)
    assert blank.max() == 0


def test_annotate_draws_something(blank):
    out = annotate(blank, [Detection("helmet", 0.9, HELMET_BOX)])
    assert out.shape == blank.shape
    assert out.max() > 0


def test_a_violation_and_a_compliant_worker_are_coloured_differently(blank):
    bare = associate_ppe_to_persons([Detection("person", 0.9, PERSON_BOX)])
    dressed = associate_ppe_to_persons(
        [
            Detection("person", 0.9, PERSON_BOX),
            Detection("helmet", 0.9, HELMET_BOX),
            Detection("vest", 0.9, (110.0, 150.0, 210.0, 280.0)),
        ]
    )
    red = annotate(blank, [], bare)
    green = annotate(blank, [], dressed)
    assert not np.array_equal(red, green)


def test_person_boxes_are_drawn_once_via_the_worker_record(blank):
    person = [Detection("person", 0.9, PERSON_BOX)]
    detections_only = annotate(blank, person)
    assert detections_only.max() == 0


def test_annotate_handles_an_empty_frame_of_detections(blank):
    assert np.array_equal(annotate(blank, []), blank)


def test_banner_writes_text(blank):
    out = draw_banner(blank.copy(), ["12.4 fps", "2 workers"])
    assert out.max() > 0
