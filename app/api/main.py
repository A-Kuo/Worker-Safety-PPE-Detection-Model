"""FastAPI demo for PPE detection and per-worker compliance.

App extras (also listed in ``app/requirements.txt``):
fastapi, uvicorn, python-multipart, streamlit, opencv-python-headless
"""

from __future__ import annotations

import base64
import os
import sys
import tempfile
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.media import (  # noqa: E402
    DEFAULT_MAX_VIDEO_FRAMES,
    DEFAULT_MAX_VIDEO_SECONDS,
    decode_image_bytes,
    encode_jpeg,
    frame_stride,
    iter_capped_frames,
    open_capture,
    open_video_writer,
    parse_stream_source,
    video_meta,
)
from app.paths import CLIP_DIR, ensure_src_on_path, resolve_weights_path  # noqa: E402
from app.runtime import IMPORT_ERROR, build_detector, default_device  # noqa: E402

ensure_src_on_path()
try:
    from ppe.compliance import Detection, WorkerCompliance, associate_ppe_to_persons  # noqa: E402
    from ppe.inference import PPEDetector  # noqa: E402
except ImportError:  # package missing; /health reports not_ready
    from app.runtime import Detection, PPEDetector, WorkerCompliance, associate_ppe_to_persons  # noqa: E402

# ---------------------------------------------------------------------------
# Detector cache (reload only when weights change; conf is updated in place)
# ---------------------------------------------------------------------------
_detector_lock = threading.Lock()
_detector = None
_detector_weights: str | None = None


def _apply_conf(detector: Any, conf: float) -> None:
    detector.conf = float(conf)


def get_detector(conf: float = 0.25, weights: str | None = None) -> PPEDetector:
    global _detector, _detector_weights
    path = str(resolve_weights_path(weights))
    with _detector_lock:
        if _detector is None or _detector_weights != path:
            _detector = build_detector(path, conf=conf, device=default_device())
            _detector_weights = path
        else:
            _apply_conf(_detector, conf)
        return _detector


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class DetectionOut(BaseModel):
    cls_name: str
    conf: float
    xyxy: list[float]


class WorkerOut(BaseModel):
    worker_id: int
    bbox: list[float]
    present: list[str]
    missing: list[str]
    violations: list[str]
    label: str


class HealthOut(BaseModel):
    status: str
    ready: bool
    weights: str
    weights_exists: bool
    ppe_package: bool
    device: str | None = None
    detail: str | None = None


class ImagePredictOut(BaseModel):
    detections: list[DetectionOut]
    compliance: list[WorkerOut]
    labels: list[str]
    annotated_jpeg_base64: str | None = None


class VideoPredictOut(BaseModel):
    source_fps: float
    frame_count: int
    frames_processed: int
    stride: int
    capped: bool
    workers_per_frame_max: int
    violation_frames: int
    violation_counts: dict[str, int]
    sample_labels: list[str]
    annotated_path: str | None = None
    annotated_url: str | None = None


def _xyxy_list(box: Any) -> list[float]:
    return [float(v) for v in box]


def detection_to_out(det: Detection) -> DetectionOut:
    return DetectionOut(
        cls_name=str(det.cls_name),
        conf=float(det.conf),
        xyxy=_xyxy_list(det.xyxy),
    )


def worker_to_out(worker: WorkerCompliance) -> WorkerOut:
    return WorkerOut(
        worker_id=int(worker.worker_id),
        bbox=_xyxy_list(worker.bbox),
        present=list(worker.present),
        missing=list(worker.missing),
        violations=list(worker.violations),
        label=str(worker.label),
    )


def _clamp_conf(conf: float) -> float:
    if conf <= 0 or conf >= 1:
        raise HTTPException(status_code=400, detail="conf must be between 0 and 1 (exclusive).")
    return float(conf)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PPE Detection API",
    description=(
        "Local demo: detect PPE, associate gear to each person, and return "
        "compliance labels such as “Worker 12 — missing helmet and vest”. "
        "GET /stream returns MJPEG; a missing camera is HTTP 503."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    return {"service": "ppe-detection", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthOut, tags=["meta"])
def health() -> HealthOut:
    weights = resolve_weights_path()
    exists = weights.is_file()
    package_ok = IMPORT_ERROR is None
    ready = package_ok and exists
    detail = None
    if not package_ok:
        detail = f"ppe import failed: {IMPORT_ERROR}"
    elif not exists:
        detail = f"Weights missing at {weights}. Set PPE_WEIGHTS."
    return HealthOut(
        status="ok" if ready else "not_ready",
        ready=ready,
        weights=str(weights),
        weights_exists=exists,
        ppe_package=package_ok,
        device=default_device(),
        detail=detail,
    )


@app.post("/predict/image", response_model=ImagePredictOut, tags=["predict"])
async def predict_image(
    file: UploadFile = File(..., description="JPEG/PNG/WebP frame"),
    conf: float = Query(0.25, description="Confidence threshold"),
    return_image: bool = Query(False, description="Include annotated JPEG as base64"),
) -> ImagePredictOut:
    """Multipart image → detections, per-worker compliance, optional annotated JPEG."""
    conf = _clamp_conf(conf)
    try:
        payload = await file.read()
        image = decode_image_bytes(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        detector = get_detector(conf=conf)
        with _detector_lock:
            detections = detector.predict_image(image)
            workers = associate_ppe_to_persons(detections)
            annotated = None
            if return_image:
                annotated, workers = detector.predict_and_comply(image)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    encoded = None
    if return_image and annotated is not None:
        encoded = base64.b64encode(encode_jpeg(annotated)).decode("ascii")

    compliance = [worker_to_out(w) for w in workers]
    return ImagePredictOut(
        detections=[detection_to_out(d) for d in detections],
        compliance=compliance,
        labels=[w.label for w in compliance],
        annotated_jpeg_base64=encoded,
    )


@app.post("/predict/video", response_model=VideoPredictOut, tags=["predict"])
async def predict_video(
    file: UploadFile = File(..., description="Video file (mp4, avi, mov, …)"),
    conf: float = Query(0.25, description="Confidence threshold"),
    return_clip: bool = Query(False, description="Write an annotated MP4 under output/app/"),
    max_frames: int = Query(
        DEFAULT_MAX_VIDEO_FRAMES,
        ge=1,
        le=2000,
        description="Hard cap on inferred frames (long clips are strided)",
    ),
    max_seconds: float = Query(
        DEFAULT_MAX_VIDEO_SECONDS,
        ge=1,
        le=300,
        description="Stop reading after this many source seconds",
    ),
) -> VideoPredictOut:
    """Upload a video, infer a capped/strided subset, return a compliance summary."""
    conf = _clamp_conf(conf)
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    tmp_path: str | None = None
    cap = None
    writer = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        cap = open_capture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open the uploaded video.")

        meta = video_meta(cap)
        source_fps = float(meta["fps"])
        frame_count = int(meta["frame_count"])
        stride = frame_stride(frame_count, max_frames)

        try:
            detector = get_detector(conf=conf)
        except (FileNotFoundError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        annotated_path: Path | None = None
        if return_clip:
            CLIP_DIR.mkdir(parents=True, exist_ok=True)
            annotated_path = CLIP_DIR / f"{uuid.uuid4().hex}.mp4"

        violation_counts: Counter[str] = Counter()
        sample_labels: list[str] = []
        seen_labels: set[str] = set()
        frames_processed = 0
        workers_max = 0
        violation_frames = 0
        size: tuple[int, int] | None = None

        for _index, frame in iter_capped_frames(
            cap, max_frames=max_frames, max_seconds=max_seconds
        ):
            if size is None:
                h, w = frame.shape[:2]
                size = (w, h)
                if annotated_path is not None:
                    writer = open_video_writer(
                        annotated_path,
                        fps=max(source_fps / stride, 1.0),
                        size=size,
                    )

            with _detector_lock:
                if return_clip:
                    annotated, workers = detector.predict_and_comply(frame)
                    if writer is not None:
                        writer.write(annotated)
                else:
                    detections = detector.predict_image(frame)
                    workers = associate_ppe_to_persons(detections)

            frames_processed += 1
            workers_max = max(workers_max, len(workers))
            frame_violation = False
            for worker in workers:
                if worker.violations or worker.missing:
                    frame_violation = True
                for item in worker.violations:
                    violation_counts[str(item)] += 1
                if worker.label not in seen_labels and len(sample_labels) < 50:
                    seen_labels.add(worker.label)
                    sample_labels.append(worker.label)
            if frame_violation:
                violation_frames += 1

        if writer is not None:
            writer.release()
            writer = None

        clip_url = None
        clip_str = None
        if annotated_path is not None and annotated_path.is_file():
            clip_str = str(annotated_path)
            clip_url = f"/clips/{annotated_path.name}"

        return VideoPredictOut(
            source_fps=source_fps,
            frame_count=frame_count,
            frames_processed=frames_processed,
            stride=stride,
            capped=frames_processed < frame_count if frame_count > 0 else False,
            workers_per_frame_max=workers_max,
            violation_frames=violation_frames,
            violation_counts=dict(violation_counts),
            sample_labels=sample_labels,
            annotated_path=clip_str,
            annotated_url=clip_url,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Video inference failed: {exc}") from exc
    finally:
        if writer is not None:
            writer.release()
        if cap is not None:
            cap.release()
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


@app.get("/clips/{name}", tags=["predict"])
def download_clip(name: str) -> FileResponse:
    """Serve an annotated clip written by ``POST /predict/video?return_clip=true``."""
    if Path(name).name != name or not name.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Invalid clip name.")
    path = CLIP_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Clip not found.")
    return FileResponse(path, media_type="video/mp4", filename=name)


@app.get("/stream", tags=["stream"])
def stream(
    source: str | None = Query(
        None,
        description="Webcam index (default 0), RTSP URL, or video file path",
    ),
    conf: float = Query(0.25, description="Confidence threshold"),
    max_fps: float = Query(12.0, ge=1, le=30, description="MJPEG send cap"),
) -> StreamingResponse:
    """MJPEG stream with boxes and compliance overlays.

    Returns **503** when the webcam index or ``source`` cannot be opened
    (typical on headless hosts with no camera).
    """
    conf = _clamp_conf(conf)
    parsed = parse_stream_source(source)
    cap = open_capture(parsed)
    if not cap.isOpened():
        cap.release()
        raise HTTPException(
            status_code=503,
            detail=(
                "No camera or media source available. "
                "Use ?source=0 for the default webcam, ?source=rtsp://host/stream "
                "for RTSP, or ?source=C:/path/video.mp4 for a file. "
                "Headless machines without a camera always return 503 for webcam indexes."
            ),
        )

    try:
        detector = get_detector(conf=conf)
    except (FileNotFoundError, RuntimeError) as exc:
        cap.release()
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    min_interval = 1.0 / float(max_fps)

    def frames() -> Any:
        try:
            while True:
                started = time.perf_counter()
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                with _detector_lock:
                    annotated, _workers = detector.predict_and_comply(frame)
                jpeg = encode_jpeg(annotated, quality=75)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
                elapsed = time.perf_counter() - started
                leftover = min_interval - elapsed
                if leftover > 0:
                    time.sleep(leftover)
        finally:
            cap.release()

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store", "Pragma": "no-cache"},
    )


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("PPE_API_HOST", "0.0.0.0")
    port = int(os.environ.get("PPE_API_PORT", "8000"))
    uvicorn.run("app.api.main:app", host=host, port=port, reload=True)
