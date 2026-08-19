"""HTTP surface, served by a pipeline with a scripted backend."""

from __future__ import annotations

import base64

import numpy as np
import pytest
from helpers import HELMET_BOX, PERSON_BOX, VEST_BOX, detection

from ppe.backends import StubBackend
from ppe.config import RuntimeConfig
from ppe.pipeline import EdgePipeline
from ppe.sources import open_writer

pytest.importorskip("fastapi", reason="the API layer needs fastapi")
pytest.importorskip("httpx", reason="TestClient needs httpx")
cv2 = pytest.importorskip("cv2", reason="the API needs opencv")

from fastapi.testclient import TestClient  # noqa: E402

from app import runtime  # noqa: E402
from app.api.main import app  # noqa: E402

BARE_HEAD = [detection("person", PERSON_BOX), detection("no_helmet", HELMET_BOX)]
DRESSED = [
    detection("person", PERSON_BOX),
    detection("helmet", HELMET_BOX),
    detection("vest", VEST_BOX),
]


@pytest.fixture
def serving():
    """Install a scripted pipeline for the duration of one test."""
    pipeline = EdgePipeline(
        StubBackend(frames=[BARE_HEAD]),
        RuntimeConfig(backend="stub", alert_min_frames=1),
    )
    runtime.set_pipeline(pipeline)
    with TestClient(app) as client:
        yield client
    runtime.clear_pipeline()


@pytest.fixture
def offline():
    runtime.clear_pipeline()
    with TestClient(app) as client:
        yield client
    runtime.clear_pipeline()


@pytest.fixture
def jpeg_bytes() -> bytes:
    image = np.full((480, 640, 3), 80, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


@pytest.fixture
def mp4_bytes(tmp_path) -> bytes:
    path = tmp_path / "in.mp4"
    writer = open_writer(path, fps=10.0, size=(64, 48))
    for _ in range(12):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def test_root(serving):
    body = serving.get("/").json()
    assert body["service"] == "ppe-detection"
    assert body["version"]


def test_classes(serving):
    body = serving.get("/classes").json()
    assert body["count"] == 14
    assert "no_helmet" in body["names"]


def test_health_is_ok_once_a_pipeline_is_loaded(serving):
    body = serving.get("/health").json()
    assert body["status"] == "ok"
    assert body["ready"] is True


def test_metrics_reports_latency(serving, jpeg_bytes):
    serving.post("/predict/image", files={"file": ("f.jpg", jpeg_bytes, "image/jpeg")})
    body = serving.get("/metrics").json()
    assert body["backend"] == "stub"
    assert body["latency"]["frames"] >= 1


def test_predict_image_returns_compliance(serving, jpeg_bytes):
    response = serving.post("/predict/image", files={"file": ("f.jpg", jpeg_bytes, "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert len(body["compliance"]) == 1
    assert body["compliance"][0]["compliant"] is False
    assert "no_helmet" in body["compliance"][0]["violations"]
    assert body["labels"][0].startswith("Worker 0:")
    assert body["latency_ms"] >= 0
    assert body["annotated_jpeg_base64"] is None


def test_predict_image_can_return_an_annotated_jpeg(serving, jpeg_bytes):
    response = serving.post(
        "/predict/image",
        params={"return_image": "true"},
        files={"file": ("f.jpg", jpeg_bytes, "image/jpeg")},
    )
    payload = response.json()["annotated_jpeg_base64"]
    decoded = base64.b64decode(payload)
    assert decoded[:2] == b"\xff\xd8"  # JPEG start of image


def test_predict_image_rejects_a_non_image(serving):
    response = serving.post("/predict/image", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400
    assert "decode" in response.json()["detail"]


def test_predict_image_validates_conf(serving, jpeg_bytes):
    response = serving.post(
        "/predict/image",
        params={"conf": 1.5},
        files={"file": ("f.jpg", jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 422


def test_stills_do_not_share_track_ids(serving, jpeg_bytes):
    for _ in range(3):
        body = serving.post(
            "/predict/image", files={"file": ("f.jpg", jpeg_bytes, "image/jpeg")}
        ).json()
        assert body["compliance"][0]["worker_id"] == 0


def test_predict_video_summarises_a_clip(serving, mp4_bytes):
    response = serving.post("/predict/video", files={"file": ("clip.mp4", mp4_bytes, "video/mp4")})
    assert response.status_code == 200
    body = response.json()
    assert body["frames_processed"] == 12
    assert body["workers_per_frame_max"] == 1
    assert body["violation_frames"] == 12
    assert body["violation_counts"]["no_helmet"] == 12
    assert body["events"]
    assert body["annotated_url"] is None


def test_predict_video_can_write_a_clip(serving, mp4_bytes):
    body = serving.post(
        "/predict/video",
        params={"return_clip": "true", "max_frames": 4},
        files={"file": ("clip.mp4", mp4_bytes, "video/mp4")},
    ).json()
    assert body["annotated_url"].startswith("/clips/")
    download = serving.get(body["annotated_url"])
    assert download.status_code == 200
    assert download.headers["content-type"] == "video/mp4"


def test_predict_video_rejects_a_non_video(serving):
    response = serving.post(
        "/predict/video", files={"file": ("clip.mp4", b"not a video", "video/mp4")}
    )
    assert response.status_code == 400


def test_clip_download_rejects_traversal(serving):
    assert serving.get("/clips/..%2Fsecret.mp4").status_code in {400, 404}


def test_clip_download_404s_for_an_unknown_name(serving):
    assert serving.get("/clips/deadbeef.mp4").status_code == 404


def test_stream_503s_without_a_source(serving):
    response = serving.get("/stream", params={"source": "/no/such/file.mp4"})
    assert response.status_code == 503
    assert "Could not open source" in response.json()["detail"]


def test_health_reports_why_it_is_not_ready(offline, monkeypatch):
    monkeypatch.setenv("PPE_WEIGHTS", "/nowhere/best.pt")
    body = offline.get("/health").json()
    assert body["ready"] is False
    assert body["weights_exists"] is False
    assert "Weights missing" in body["detail"]


def test_predict_image_503s_without_a_usable_model(offline, jpeg_bytes, monkeypatch):
    monkeypatch.setenv("PPE_WEIGHTS", "/nowhere/best.pt")
    response = offline.post("/predict/image", files={"file": ("f.jpg", jpeg_bytes, "image/jpeg")})
    assert response.status_code == 503
