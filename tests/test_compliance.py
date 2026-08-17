"""Person–PPE association on synthetic boxes."""

from __future__ import annotations

from ppe.compliance import (
    NEGATIVE_VIOLATIONS,
    REQUIRED_ON_PERSON,
    Detection,
    associate_ppe_to_persons,
)


def test_required_and_negative_constants():
    assert REQUIRED_ON_PERSON == ("helmet", "vest")
    assert NEGATIVE_VIOLATIONS == (
        "no_helmet",
        "no_vest",
        "no_goggles",
        "no_gloves",
        "no_mask",
    )


def test_helmet_inside_person_is_present():
    person = Detection("person", 0.95, (0.0, 0.0, 100.0, 200.0))
    helmet = Detection("helmet", 0.90, (20.0, 0.0, 60.0, 40.0))
    workers = associate_ppe_to_persons([person, helmet])
    assert len(workers) == 1
    worker = workers[0]
    assert worker.worker_id == 0
    assert worker.bbox == person.xyxy
    assert worker.present == ["helmet"]
    assert worker.missing == ["vest"]
    assert "vest" in worker.violations
    assert "helmet" not in worker.missing
    assert "missing vest" in worker.label


def test_bare_person_missing_helmet_and_vest():
    person = Detection("person", 0.9, (0.0, 0.0, 80.0, 160.0))
    workers = associate_ppe_to_persons([person])
    assert workers[0].present == []
    assert workers[0].missing == ["helmet", "vest"]
    assert workers[0].violations == ["helmet", "vest"]
    assert workers[0].label == "Worker 0 — missing helmet and vest"


def test_no_helmet_box_is_a_violation():
    person = Detection("person", 0.9, (0.0, 0.0, 100.0, 200.0))
    no_helmet = Detection("no_helmet", 0.8, (25.0, 5.0, 55.0, 35.0))
    worker = associate_ppe_to_persons([person, no_helmet])[0]
    assert "no_helmet" in worker.violations
    assert "helmet" in worker.missing
    assert "helmet" not in worker.present


def test_compliant_worker_label():
    person = Detection("person", 0.9, (0.0, 0.0, 100.0, 200.0))
    helmet = Detection("helmet", 0.9, (20.0, 0.0, 60.0, 40.0))
    vest = Detection("vest", 0.85, (15.0, 50.0, 85.0, 140.0))
    worker = associate_ppe_to_persons([person, helmet, vest])[0]
    assert worker.present == ["helmet", "vest"]
    assert worker.missing == []
    assert worker.violations == []
    assert worker.label == "Worker 0 — compliant"


def test_distant_ppe_is_not_associated():
    person = Detection("person", 0.9, (0.0, 0.0, 50.0, 50.0))
    helmet = Detection("helmet", 0.8, (200.0, 200.0, 250.0, 250.0))
    worker = associate_ppe_to_persons([person, helmet], iou_thresh=0.05)[0]
    assert "helmet" not in worker.present
    assert "helmet" in worker.missing


def test_iou_associates_when_center_is_outside():
    person = Detection("person", 0.9, (0.0, 0.0, 100.0, 100.0))
    # Center (110, 10) is outside the person; IoU is ~0.0625 > 0.05.
    vest = Detection("vest", 0.8, (80.0, -20.0, 140.0, 40.0))
    worker = associate_ppe_to_persons([person, vest], iou_thresh=0.05)[0]
    assert "vest" in worker.present


def test_iou_below_threshold_without_containment_is_ignored():
    person = Detection("person", 0.9, (0.0, 0.0, 100.0, 100.0))
    vest = Detection("vest", 0.8, (80.0, -20.0, 140.0, 40.0))
    worker = associate_ppe_to_persons([person, vest], iou_thresh=0.5)[0]
    assert "vest" not in worker.present


def test_ppe_assigned_to_containing_person_not_neighbor():
    left = Detection("person", 0.9, (0.0, 0.0, 100.0, 200.0))
    right = Detection("person", 0.9, (150.0, 0.0, 250.0, 200.0))
    helmet = Detection("helmet", 0.9, (170.0, 10.0, 210.0, 50.0))
    workers = associate_ppe_to_persons([left, right, helmet])
    assert workers[0].present == []
    assert workers[1].present == ["helmet"]
    assert workers[1].worker_id == 1


def test_no_persons_returns_empty():
    helmet = Detection("helmet", 0.8, (0.0, 0.0, 10.0, 10.0))
    assert associate_ppe_to_persons([helmet]) == []


def test_scene_classes_are_not_associated_as_ppe():
    person = Detection("person", 0.9, (0.0, 0.0, 100.0, 200.0))
    cone = Detection("cone", 0.9, (10.0, 10.0, 40.0, 80.0))
    worker = associate_ppe_to_persons([person, cone])[0]
    assert worker.present == []
    assert "cone" not in worker.violations


def test_person_name_is_case_insensitive():
    person = Detection("Person", 0.9, (0.0, 0.0, 100.0, 200.0))
    helmet = Detection("helmet", 0.9, (20.0, 0.0, 60.0, 40.0))
    workers = associate_ppe_to_persons([person, helmet])
    assert len(workers) == 1
    assert "helmet" in workers[0].present
