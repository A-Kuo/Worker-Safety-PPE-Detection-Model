"""The per-frame path, driven by a scripted backend."""

from __future__ import annotations

import pytest
from helpers import HELMET_BOX, PERSON_BOX, SECOND_PERSON_BOX, VEST_BOX, detection, shifted

from ppe.backends import RawDetection, StubBackend
from ppe.config import RuntimeConfig
from ppe.pipeline import EdgePipeline, LatencyStats, summarize_run


def test_a_compliant_frame_raises_nothing(make_pipeline, frame, compliant_frame):
    pipeline = make_pipeline([compliant_frame])
    result = pipeline.process(frame)
    assert len(result.workers) == 1
    assert result.workers[0].compliant
    assert result.events == []


def test_a_bare_head_is_flagged(make_pipeline, frame, bare_head_frame):
    pipeline = make_pipeline([bare_head_frame], alert_min_frames=1)
    result = pipeline.process(frame)
    assert "no_helmet" in result.workers[0].violations
    assert sorted(event.violation for event in result.events) == ["helmet", "no_helmet"]


def test_raw_class_names_are_unified(frame):
    # A Construction-vocabulary checkpoint reports "Person" and "Hardhat".
    backend = StubBackend(
        frames=[
            [
                RawDetection(5, "Person", 0.9, PERSON_BOX),
                RawDetection(0, "Hardhat", 0.9, HELMET_BOX),
            ]
        ]
    )
    result = EdgePipeline(backend, RuntimeConfig(backend="stub")).process(frame)
    assert sorted(d.cls_name for d in result.detections) == ["helmet", "person"]
    assert result.workers[0].present == ["helmet"]


def test_worker_ids_persist_across_frames(make_pipeline, frame):
    frames = [
        [detection("person", PERSON_BOX)],
        [detection("person", shifted(PERSON_BOX, dx=6))],
        [detection("person", shifted(PERSON_BOX, dx=12))],
    ]
    pipeline = make_pipeline(frames)
    assert [pipeline.process(frame).workers[0].worker_id for _ in range(3)] == [0, 0, 0]


def test_alerts_wait_for_the_configured_streak(make_pipeline, frame, clock):
    frames = [[detection("person", PERSON_BOX)]]
    pipeline = make_pipeline(frames, alert_min_frames=3, alert_cooldown_s=100.0)
    pipeline.clock = clock
    raised = []
    for _ in range(5):
        clock.advance(0.1)
        raised.extend(pipeline.process(frame).events)
    # helmet and vest each raise exactly once inside the cooldown.
    assert sorted(event.violation for event in raised) == ["helmet", "vest"]


def test_stride_skips_frames(make_pipeline, frame, compliant_frame):
    pipeline = make_pipeline([compliant_frame], frame_stride=3)
    results = [pipeline.process(frame) for _ in range(6)]
    assert [r.skipped for r in results] == [False, True, True, False, True, True]
    assert pipeline.backend.calls == 2


def test_process_stream_drops_skipped_frames(make_pipeline, frame, compliant_frame):
    pipeline = make_pipeline([compliant_frame], frame_stride=2)
    results = list(pipeline.process_stream([frame] * 6))
    assert len(results) == 3
    assert all(not r.skipped for r in results)


def test_latency_is_recorded(make_pipeline, frame, compliant_frame):
    pipeline = make_pipeline([compliant_frame])
    for _ in range(4):
        pipeline.process(frame)
    stats = pipeline.stats()
    assert stats["latency"]["frames"] == 4
    assert stats["latency"]["mean_ms"] >= 0.0
    assert stats["backend"] == "stub"


def test_reset_clears_state(make_pipeline, frame, bare_head_frame):
    pipeline = make_pipeline([bare_head_frame], alert_min_frames=1)
    pipeline.process(frame)
    pipeline.reset()
    assert pipeline.stats()["frames"] == 0
    assert pipeline.stats()["tracks_open"] == 0
    assert pipeline.process(frame).workers[0].worker_id == 0


def test_two_people_get_distinct_ids(make_pipeline, frame):
    frames = [[detection("person", PERSON_BOX), detection("person", SECOND_PERSON_BOX)]]
    workers = make_pipeline(frames).process(frame).workers
    assert sorted(worker.worker_id for worker in workers) == [0, 1]


def test_required_ppe_config_reaches_compliance(make_pipeline, frame):
    frames = [[detection("person", PERSON_BOX), detection("vest", VEST_BOX)]]
    pipeline = make_pipeline(frames, required_ppe=("vest",))
    assert pipeline.process(frame).workers[0].compliant


def test_warmup_does_not_count_as_a_frame(make_pipeline, frame, compliant_frame):
    pipeline = make_pipeline([compliant_frame])
    pipeline.warmup(frame, rounds=3)
    assert pipeline.stats()["frames"] == 0
    assert pipeline.backend.calls == 3


def test_frame_result_serialises(make_pipeline, frame, bare_head_frame):
    result = make_pipeline([bare_head_frame], alert_min_frames=1).process(frame)
    payload = result.as_dict()
    assert payload["index"] == 0
    assert payload["compliance"][0]["violations"]
    assert isinstance(payload["events"][0], str)


def test_violation_count_sums_across_workers(make_pipeline, frame):
    frames = [[detection("person", PERSON_BOX), detection("person", SECOND_PERSON_BOX)]]
    result = make_pipeline(frames).process(frame)
    assert result.violation_count == 4


def test_summarize_run_aggregates(make_pipeline, frame, compliant_frame, bare_head_frame):
    pipeline = make_pipeline([compliant_frame, bare_head_frame], alert_min_frames=1)
    results = [pipeline.process(frame) for _ in range(4)]
    report = summarize_run(results)
    assert report["frames"] == 4
    assert report["workers_max"] == 1
    assert report["violation_frames"] == 2
    assert report["violation_counts"]["no_helmet"] == 2


def test_context_manager_closes_the_backend(make_pipeline, frame, compliant_frame):
    pipeline = make_pipeline([compliant_frame])
    with pipeline:
        pipeline.process(frame)
    assert pipeline.backend.calls == 0


def test_latency_percentiles():
    stats = LatencyStats()
    for value in range(1, 101):
        stats.add(float(value))
    assert stats.percentile(0.5) == pytest.approx(50.0, abs=1.0)
    assert stats.percentile(0.95) == pytest.approx(95.0, abs=1.0)
    assert stats.as_dict()["frames"] == 100


def test_empty_latency_window():
    assert LatencyStats().as_dict() == {"frames": 0}
