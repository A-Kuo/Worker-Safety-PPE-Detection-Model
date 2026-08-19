"""Identity stability across frames."""

from __future__ import annotations

import pytest
from helpers import PERSON_BOX, SECOND_PERSON_BOX, shifted

from ppe.tracking import IouTracker, iou


def test_iou_of_identical_boxes_is_one():
    assert iou(PERSON_BOX, PERSON_BOX) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0


def test_iou_of_touching_boxes_is_zero():
    assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_iou_of_a_half_overlap():
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_first_frame_assigns_sequential_ids():
    tracker = IouTracker()
    tracks = tracker.update([PERSON_BOX, SECOND_PERSON_BOX])
    assert [track.track_id for track in tracks] == [0, 1]


def test_slow_movement_keeps_the_same_id():
    tracker = IouTracker(iou_threshold=0.3)
    first = tracker.update([PERSON_BOX])[0]
    for step in range(1, 6):
        moved = tracker.update([shifted(PERSON_BOX, dx=4 * step)])[0]
        assert moved.track_id == first.track_id
    assert moved.hits == 6


def test_a_teleporting_box_gets_a_new_id():
    tracker = IouTracker(iou_threshold=0.3)
    first = tracker.update([PERSON_BOX])[0]
    second = tracker.update([shifted(PERSON_BOX, dx=1000)])[0]
    assert second.track_id != first.track_id


def test_two_people_do_not_swap_ids():
    tracker = IouTracker()
    tracker.update([PERSON_BOX, SECOND_PERSON_BOX])
    # Report them in the opposite order on the next frame.
    tracks = tracker.update([shifted(SECOND_PERSON_BOX, dy=5), shifted(PERSON_BOX, dy=5)])
    assert [track.track_id for track in tracks] == [1, 0]


def test_a_track_survives_a_brief_dropout():
    tracker = IouTracker(max_age=5)
    first = tracker.update([PERSON_BOX])[0]
    for _ in range(3):
        tracker.update([])
    again = tracker.update([PERSON_BOX])[0]
    assert again.track_id == first.track_id


def test_a_track_expires_after_max_age():
    tracker = IouTracker(max_age=2)
    tracker.update([PERSON_BOX])
    for _ in range(4):
        tracker.update([])
    assert tracker.tracks == []
    assert tracker.update([PERSON_BOX])[0].track_id == 1


def test_history_is_bounded():
    tracker = IouTracker(history_length=4)
    for step in range(10):
        tracker.update([shifted(PERSON_BOX, dx=step)])
    assert len(tracker.tracks[0].history) == 4


def test_centroid_is_the_box_center():
    tracker = IouTracker()
    track = tracker.update([(0.0, 0.0, 10.0, 20.0)])[0]
    assert track.centroid == (5.0, 10.0)


def test_reset_clears_ids():
    tracker = IouTracker()
    tracker.update([PERSON_BOX])
    tracker.reset()
    assert tracker.update([PERSON_BOX])[0].track_id == 0


def test_confidence_length_must_match():
    with pytest.raises(ValueError, match="same length"):
        IouTracker().update([PERSON_BOX], [0.9, 0.8])


def test_threshold_must_be_a_fraction():
    with pytest.raises(ValueError, match="iou_threshold"):
        IouTracker(iou_threshold=1.5)
