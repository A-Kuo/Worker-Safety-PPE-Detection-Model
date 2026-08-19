"""Alert debouncing: persistence before raising, cooldown after."""

from __future__ import annotations

import pytest

from ppe.compliance import WorkerCompliance
from ppe.events import ViolationMonitor


def worker(worker_id: int, *violations: str) -> WorkerCompliance:
    return WorkerCompliance(
        worker_id=worker_id,
        bbox=(0.0, 0.0, 10.0, 20.0),
        present=[],
        missing=list(violations),
        violations=list(violations),
        label=f"Worker {worker_id}",
    )


def test_a_single_frame_does_not_raise():
    monitor = ViolationMonitor(min_frames=3)
    assert monitor.update([worker(0, "helmet")], 0.0) == []


def test_raises_once_the_streak_is_long_enough():
    monitor = ViolationMonitor(min_frames=3)
    events = []
    for frame in range(3):
        events += monitor.update([worker(0, "helmet")], frame * 0.1)
    assert len(events) == 1
    assert events[0].track_id == 0
    assert events[0].violation == "helmet"
    assert events[0].frames == 3
    assert events[0].repeat is False


def test_cooldown_suppresses_repeats():
    monitor = ViolationMonitor(min_frames=1, cooldown_s=10.0)
    first = monitor.update([worker(0, "helmet")], 0.0)
    quiet = [monitor.update([worker(0, "helmet")], t) for t in (1.0, 5.0, 9.9)]
    assert len(first) == 1
    assert all(batch == [] for batch in quiet)


def test_a_repeat_raises_after_the_cooldown():
    monitor = ViolationMonitor(min_frames=1, cooldown_s=10.0)
    monitor.update([worker(0, "helmet")], 0.0)
    later = monitor.update([worker(0, "helmet")], 10.5)
    assert len(later) == 1
    assert later[0].repeat is True
    assert later[0].duration_s == pytest.approx(10.5)


def test_each_worker_is_tracked_separately():
    monitor = ViolationMonitor(min_frames=1)
    events = monitor.update([worker(0, "helmet"), worker(1, "vest")], 0.0)
    assert {(e.track_id, e.violation) for e in events} == {(0, "helmet"), (1, "vest")}


def test_each_violation_type_raises_separately():
    monitor = ViolationMonitor(min_frames=1)
    events = monitor.update([worker(0, "helmet", "vest")], 0.0)
    assert sorted(e.violation for e in events) == ["helmet", "vest"]


def test_a_gap_resets_the_streak():
    monitor = ViolationMonitor(min_frames=3, forget_after_s=5.0)
    monitor.update([worker(0, "helmet")], 0.0)
    monitor.update([worker(0, "helmet")], 0.1)
    monitor.update([], 0.2)
    assert monitor.update([worker(0, "helmet")], 0.3) == []
    assert monitor.update([worker(0, "helmet")], 0.4) == []
    assert len(monitor.update([worker(0, "helmet")], 0.5)) == 1


def test_state_is_dropped_once_a_violation_is_long_gone():
    monitor = ViolationMonitor(min_frames=1, forget_after_s=2.0)
    monitor.update([worker(0, "helmet")], 0.0)
    monitor.update([], 5.0)
    assert monitor.active() == []


def test_active_lists_debounced_violations():
    monitor = ViolationMonitor(min_frames=2)
    monitor.update([worker(4, "vest")], 0.0)
    assert monitor.active() == []
    monitor.update([worker(4, "vest")], 0.1)
    assert monitor.active() == [(4, "vest")]


def test_describe_reads_as_a_log_line():
    monitor = ViolationMonitor(min_frames=1)
    event = monitor.update([worker(7, "helmet")], 3.0)[0]
    assert event.describe().startswith("worker 7 now helmet")


def test_reset_clears_all_streaks():
    monitor = ViolationMonitor(min_frames=2)
    monitor.update([worker(0, "helmet")], 0.0)
    monitor.reset()
    assert monitor.update([worker(0, "helmet")], 0.1) == []


def test_min_frames_must_be_positive():
    with pytest.raises(ValueError, match="min_frames"):
        ViolationMonitor(min_frames=0)
