"""Live window (PR 3): frame slot, pipeline threads, HUD honesty, CLI. GUI tests run offscreen."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from ppe.compliance import Detection
from ppe.runtime.config import ExecutionPolicy
from ppe.window.pipeline import (
    ImageSource,
    LatestSlot,
    Pipeline,
    SyntheticSource,
    device_label,
    draw_hud,
    hud_lines,
    open_source,
)

REPO = Path(__file__).resolve().parents[1]


# --- LatestSlot -----------------------------------------------------------------------------------

def test_slot_keeps_only_the_newest_and_counts_replacements():
    slot = LatestSlot()
    for i in range(5):
        slot.put(i)
    assert slot.replaced == 4
    assert slot.take(timeout=0) == 4
    assert slot.take(timeout=0) is None  # empty: returns, never blocks forever


def test_slot_take_waits_for_a_producer():
    slot = LatestSlot()
    threading.Timer(0.05, slot.put, args=("x",)).start()
    assert slot.take(timeout=2) == "x"


def test_slot_lossless_put_blocks_until_taken_and_respects_stop():
    slot = LatestSlot()
    assert slot.put_when_empty(1, lambda: False) is True
    done = []
    t = threading.Thread(target=lambda: done.append(slot.put_when_empty(2, lambda: False)))
    t.start()
    time.sleep(0.1)
    assert done == []  # still waiting: nothing was overwritten
    assert slot.take(timeout=1) == 1
    t.join(2)
    assert done == [True] and slot.take(timeout=1) == 2 and slot.replaced == 0
    slot.put(3)
    assert slot.put_when_empty(4, lambda: True) is False  # stop requested while full


def test_closed_slot_hands_out_the_last_item_then_none():
    slot = LatestSlot()
    slot.put("last")
    slot.close()
    assert slot.closed and slot.take(timeout=0) == "last" and slot.take(timeout=0) is None
    assert slot.put_when_empty("x", lambda: False) is False


# --- sources ----------------------------------------------------------------------------------------

def test_synthetic_source_yields_n_distinct_frames_then_ends():
    src = SyntheticSource(frames=3, size=(64, 48))
    a, b, c = src.read(), src.read(), src.read()
    assert a.shape == (48, 64, 3) and a.dtype == np.uint8
    assert not np.array_equal(a, b) and src.read() is None and c is not None


def test_open_source_dispatch(tmp_path):
    cv2 = pytest.importorskip("cv2")
    assert isinstance(open_source("synthetic", frames=1), SyntheticSource)
    img = tmp_path / "f.png"
    cv2.imwrite(str(img), np.zeros((10, 12, 3), dtype=np.uint8))
    src = open_source(str(img), frames=2)
    assert isinstance(src, ImageSource) and not src.live
    assert src.read().shape == (10, 12, 3) and src.read() is not None and src.read() is None
    with pytest.raises(RuntimeError, match="Could not read image"):
        open_source(str(tmp_path / "missing.png"))
    with pytest.raises(RuntimeError, match="Could not open video source"):
        open_source(str(tmp_path / "missing.mp4"))


def test_capture_helpers_are_importable_from_the_package_and_the_demo():
    from app import media
    from ppe.io import capture

    for name in ("open_capture", "read_latest_frame", "configure_capture_for_latency",
                 "iter_capped_frames", "open_video_writer", "parse_stream_source"):
        assert getattr(media, name) is getattr(capture, name)
    assert capture.parse_stream_source(None) == 0 and capture.parse_stream_source("2") == 2
    assert capture.parse_stream_source("rtsp://x/y") == "rtsp://x/y"


# --- pipeline ---------------------------------------------------------------------------------------

class _Session:
    """Fake session: one vest box per frame, optional delay, records what it saw."""

    def __init__(self, delay=0.0, fail_at=None):
        self.policy = ExecutionPolicy(conf=0.30)
        self.delay, self.fail_at = delay, fail_at
        self.seen = 0
        self.threads: set[str] = set()
        self.last_timings = {}

    def predict(self, frame):
        self.seen += 1
        self.threads.add(threading.current_thread().name)
        if self.fail_at and self.seen >= self.fail_at:
            raise RuntimeError("device lost")
        if self.delay:
            time.sleep(self.delay)
        self.last_timings = {"preprocess_ms": 1.0, "infer_ms": 4.0, "postprocess_ms": 0.5, "total_ms": 5.5}
        return [Detection("vest", 0.9, (2.0, 2.0, 20.0, 20.0))]


class _LiveSource:
    """Camera-like: produces frames on its own clock regardless of the consumer."""

    live = True
    label = "fake camera"

    def __init__(self, frames, interval=0.002):
        self.left, self.interval = frames, interval

    def read(self):
        if self.left <= 0:
            return None
        self.left -= 1
        time.sleep(self.interval)
        return np.zeros((48, 64, 3), dtype=np.uint8)

    def close(self):
        pass


def _drain(pipeline, timeout=10.0):
    out, end = [], time.time() + timeout
    while time.time() < end:
        r = pipeline.results.take(timeout=0.05)
        if r is not None:
            out.append(r)
        elif pipeline.finished.is_set():
            break
    return out


def test_lossless_pipeline_processes_every_frame_in_order_off_the_caller_thread():
    session = _Session()
    p = Pipeline(session, SyntheticSource(frames=25, size=(64, 48)), lossless=True, pace=False).start()
    results = _drain(p)
    p.stop()
    assert [r.index for r in results] == list(range(1, 26))
    assert session.seen == 25 and p.stats.snapshot()["dropped"] == 0
    assert session.threads == {"ppe-infer"}
    assert results[0].counts == {"vest": 1} and results[0].timings["infer_ms"] == 4.0
    assert results[0].image.shape == (48, 64, 3)


def test_live_pipeline_drops_stale_frames_instead_of_falling_behind():
    session = _Session(delay=0.03)  # inference ~10x slower than the camera
    p = Pipeline(session, _LiveSource(frames=60)).start()
    results = _drain(p)
    p.stop()
    snap = p.stats.snapshot()
    assert snap["captured"] == 60
    assert 0 < session.seen < 60 and snap["dropped"] > 0  # skipped, not queued
    indices = [r.index for r in results]
    assert indices == sorted(indices) and indices[-1] > session.seen  # showing recent frames


def test_worker_error_is_surfaced_not_swallowed():
    p = Pipeline(_Session(fail_at=3), SyntheticSource(frames=50, size=(64, 48)), lossless=True, pace=False).start()
    _drain(p, timeout=5)
    assert p.finished.is_set() and isinstance(p.error, RuntimeError) and "device lost" in str(p.error)
    p.stop()


def test_pause_stops_capture_and_resume_continues():
    session = _Session()
    p = Pipeline(session, _LiveSource(frames=10_000, interval=0.001)).start()
    time.sleep(0.15)
    assert p.toggle_pause() is True
    time.sleep(0.1)
    frozen = p.stats.snapshot()["captured"]
    time.sleep(0.15)
    assert p.stats.snapshot()["captured"] == frozen
    assert p.toggle_pause() is False
    time.sleep(0.15)
    assert p.stats.snapshot()["captured"] > frozen
    p.stop()


def test_threshold_adjustment_is_clamped_and_applies_to_the_session_policy():
    session = _Session()
    p = Pipeline(session, SyntheticSource(frames=1))
    assert p.adjust_conf(+0.05) == 0.35 and session.policy.conf == 0.35
    for _ in range(40):
        p.adjust_conf(+0.05)
    assert session.policy.conf == 0.95
    for _ in range(40):
        p.adjust_conf(-0.05)
    assert session.policy.conf == 0.05


def test_recording_from_start_includes_the_first_frame(tmp_path):
    cv2 = pytest.importorskip("cv2")
    out = tmp_path / "clip.mp4"
    p = Pipeline(_Session(), SyntheticSource(frames=12, size=(64, 48)), lossless=True, pace=False, record=out).start()
    _drain(p)
    p.stop()
    cap = cv2.VideoCapture(str(out))
    if not cap.isOpened():
        pytest.skip("mp4v writer unavailable in this OpenCV build")
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 12  # none lost at the start
    cap.release()
    assert p.recording_path is None  # stop() closed the file


def test_paced_file_source_is_not_read_faster_than_its_frame_rate():
    class _Slow(SyntheticSource):
        fps = 50.0

    p = Pipeline(_Session(), _Slow(frames=10, size=(64, 48)), lossless=True, pace=True)
    t0 = time.perf_counter()
    p.start()
    _drain(p)
    p.stop()
    assert time.perf_counter() - t0 >= 0.15  # ~9 intervals of 20 ms


# --- HUD honesty ------------------------------------------------------------------------------------

OV, CPU = "OpenVINOExecutionProvider", "CPUExecutionProvider"


def _info(active, device=None, strict=False, missing=()):
    return {"backend": {"providers_active": active, "providers_missing": list(missing),
                        "openvino_device_type": device, "strict_npu": strict}}


def test_device_label_never_treats_the_always_present_cpu_ep_as_the_device():
    assert device_label(_info([OV, CPU], "NPU", strict=True)) == "OpenVINO:NPU strict"


def test_device_label_reports_verified_unverified_and_cpu_fallback():
    ok = {"ok": True, "npu_fraction": 1.0, "cpu_resident": []}
    assert device_label(_info([OV, CPU], "NPU", True), ok).endswith("verified on NPU (100% of nodes)")
    gpu = {"ok": False, "cpu_resident": []}
    assert "not pinned to the NPU" in device_label(_info([OV, CPU], "GPU"), gpu)
    cpu = {"ok": False, "cpu_resident": [{"node": "a", "op": "Conv"}] * 3}
    assert "NOT NPU-only (3 CPU nodes)" in device_label(_info([CPU]), cpu)
    assert "placement unknown" in device_label(_info([OV, CPU], "NPU"), {"ok": None})
    # Requested OpenVINO but only the CPU EP is active: say so plainly.
    assert device_label(_info([CPU], missing=[OV])) == "CPU (OpenVINO did not load)"
    assert device_label(_info([CPU])) == "CPU"  # a CPU run that was asked for is just "CPU"
    assert device_label(_info([OV, CPU])) == "OpenVINO:default device"


def test_hud_lines_content():
    snap = {"fps": 31.26, "dropped": 2, "latency_ms": {"n": 10, "p50": 5.31, "p95": 7.0},
            "stages_p50_ms": {"preprocess_ms": 1.3, "infer_ms": 2.6, "postprocess_ms": 0.3}}
    lines = hud_lines(snap, {"vest": 3, "no_vest": 1}, device="OpenVINO:NPU strict", conf=0.3,
                      paused=True, recording=True)
    assert lines[0] == "OpenVINO:NPU strict"
    assert "31.3 FPS" in lines[1] and "p50 5.3 ms" in lines[1] and "p95 7.0 ms" in lines[1]
    assert "pre 1.3" in lines[2] and "infer 2.6" in lines[2] and "post 0.3" in lines[2]
    assert "vest 3" in lines[3] and "no_vest 1" in lines[3] and "conf 0.30" in lines[3] and "dropped 2" in lines[3]
    assert lines[4] == "PAUSED  REC"
    # Before any frame was timed there are no fake numbers.
    early = hud_lines({"fps": 0, "latency_ms": {"n": 0}}, {}, device="d", conf=0.3)
    assert early == ["d", "0.0 FPS", "vest 0   no_vest 0   conf 0.30   dropped 0"]


def test_draw_hud_modifies_only_the_band_and_survives_tiny_frames():
    pytest.importorskip("cv2")
    img = np.full((200, 400, 3), 200, dtype=np.uint8)
    draw_hud(img, ["hello", "world"])
    assert img[:20, :60].mean() < 200 and (img[150:, :] == 200).all()
    tiny = np.full((8, 8, 3), 200, dtype=np.uint8)
    draw_hud(tiny, ["a very long line that does not fit"])  # must not raise
    untouched = np.full((20, 20, 3), 7, dtype=np.uint8)
    assert (draw_hud(untouched, []) == 7).all()


# --- CLI / identity ---------------------------------------------------------------------------------

def test_window_without_pyside6_names_the_extra(monkeypatch):
    from ppe import cli

    monkeypatch.setitem(sys.modules, "PySide6", None)
    with pytest.raises(SystemExit, match=r"ppe\[gui\]"):
        cli.main(["window", "--selftest", "--source", "synthetic"])


def test_gui_extra_is_declared_and_core_stays_gui_free():
    tomllib = pytest.importorskip("tomllib")
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert any(d.startswith("PySide6") for d in data["project"]["optional-dependencies"]["gui"])
    assert not any("PySide6" in d for d in data["project"]["dependencies"])
    assert "opencv-python-headless" in " ".join(data["project"]["dependencies"])


def test_pipeline_module_has_no_gui_import():
    src = (REPO / "src" / "ppe" / "window" / "pipeline.py").read_text(encoding="utf-8")
    assert "PySide6" not in src and "imshow" not in src


def test_about_carries_licence_and_source_and_the_api_root_exposes_them():
    from ppe import about

    assert about.LICENSE == "AGPL-3.0" and about.SOURCE_URL.startswith("https://github.com/")
    assert about.SOURCE_URL in about.notice()
    pytest.importorskip("fastapi")
    from app.api.main import root

    body = root()
    assert body["license"] == "AGPL-3.0" and body["source"] == about.SOURCE_URL


def test_readme_does_not_tell_users_to_bind_all_interfaces():
    assert "--host 0.0.0.0" not in (REPO / "README.md").read_text(encoding="utf-8")


# --- real Qt, offscreen -----------------------------------------------------------------------------

@pytest.fixture
def offscreen(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")


class _InfoSession(_Session):
    model_path = Path("m.onnx")

    class backend:  # noqa: N801 - attribute shaped like the real session's backend
        name = "fake"

    def info(self):
        return _info([OV, CPU], "NPU", strict=True)


def test_window_selftest_displays_frames_and_writes_a_snapshot(offscreen, tmp_path):
    cv2 = pytest.importorskip("cv2")
    from ppe.window.app import run_window

    snap = tmp_path / "shot.png"
    summary = run_window(_InfoSession(), SyntheticSource(frames=30, size=(160, 120)), selftest_frames=20,
                         verify=False, snapshot=snap, out_dir=tmp_path)
    assert summary["ok"] is True and summary["error"] is None
    assert summary["frames_shown"] == 20 and summary["frames_dropped"] == 0
    assert summary["device"] == "OpenVINO:NPU strict" and summary["placement_ok"] is None
    assert summary["last_counts"] == {"vest": 1} and summary["display_fps"] > 0
    json.dumps(summary)
    shot = cv2.imread(str(snap))
    assert shot is not None and shot.shape == (120, 160, 3)


def test_window_closes_itself_when_the_source_ends(offscreen, tmp_path):
    from ppe.window.app import run_window

    summary = run_window(_InfoSession(), SyntheticSource(frames=5, size=(64, 48)), selftest_frames=999,
                         verify=False, out_dir=tmp_path)
    assert summary["frames_shown"] == 5  # ended with the stream, did not hang waiting for 999


def test_window_reports_a_worker_failure_and_exits(offscreen, tmp_path):
    from ppe.window.app import run_window

    summary = run_window(_InfoSession(fail_at=4), SyntheticSource(frames=50, size=(64, 48)),
                         selftest_frames=40, verify=False, out_dir=tmp_path)
    assert summary["ok"] is False and "device lost" in summary["error"]


def test_cli_window_selftest_end_to_end_with_real_onnxruntime(offscreen, tmp_path, capsys):
    pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    from ppe import cli

    x = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])
    y = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 6, 512])
    # (1,3,32,32) -> (1,6,512): a real node for the profiler; all-zero scores mean no detections.
    shape = helper.make_tensor("shape", TensorProto.INT64, [3], [1, 6, 512])
    zero = helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0])
    nodes = [helper.make_node("Mul", ["images", "zero"], ["z"], name="mul0"),
             helper.make_node("Reshape", ["z", "shape"], ["out"], name="reshape0")]
    graph = helper.make_graph(nodes, "g", [x], [y], initializer=[shape, zero])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "tiny.onnx"
    onnx.save(model, str(path))

    code = cli.main(["window", "--selftest", "--frames", "8", "--source", "synthetic", "--imgsz", "32",
                     "--weights", str(path), "--providers", "cpu"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["frames_shown"] == 8 and out["strict_npu"] is False
    assert out["device"].startswith("CPU") and out["placement_ok"] is False  # CPU is never called NPU

    with pytest.raises(SystemExit, match="ppe window:"):
        cli.main(["window", "--selftest", "--source", "synthetic", "--weights", str(path), "--strict-npu",
                  "--providers", "cpu"])
    assert os.environ.get("QT_QPA_PLATFORM") == "offscreen"
