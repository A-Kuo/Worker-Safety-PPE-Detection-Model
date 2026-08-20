"""HTTP service around the PPE pipeline.

Endpoints cover the three shapes a site integration tends to need: score one
frame, score an uploaded clip, or pull a live MJPEG preview. Extra deps for
this layer are in ``app/requirements.txt``.
"""

from __future__ import annotations

import base64
import os
import sys
import tempfile
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.media import (  # noqa: E402
    DEFAULT_MAX_VIDEO_FRAMES,
    DEFAULT_MAX_VIDEO_SECONDS,
    decode_image_bytes,
    encode_jpeg,
    iter_capped_frames,
)
from app.paths import CLIP_DIR, ensure_src_on_path  # noqa: E402
from app.runtime import (  # noqa: E402
    PIPELINE_LOCK,
    get_pipeline,
    runtime_status,
)

ensure_src_on_path()

from ppe import __version__  # noqa: E402
from ppe.draw import annotate  # noqa: E402
from ppe.providers import describe_environment  # noqa: E402
from ppe.schema import UNIFIED_CLASS_NAMES  # noqa: E402
from ppe.sources import (  # noqa: E402
    SourceUnavailable,
    capture_info,
    frame_stride,
    open_capture,
    open_writer,
)


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
    compliant: bool


class HealthOut(BaseModel):
    status: str
    ready: bool
    version: str
    weights: str
    weights_exists: bool
    backend: str
    execution: str
    provider: str | None = None
    npu_available: list[str] = Field(default_factory=list)
    device: str | None = None
    detail: str | None = None


class ImagePredictOut(BaseModel):
    detections: list[DetectionOut]
    compliance: list[WorkerOut]
    labels: list[str]
    latency_ms: float
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
    events: list[str] = Field(default_factory=list)
    latency: dict[str, float] = Field(default_factory=dict)
    annotated_path: str | None = None
    annotated_url: str | None = None


app = FastAPI(
    title="PPE Detection API",
    description=(
        "Detect PPE, attach it to each person in frame, and report who is out of "
        "compliance. GET /stream returns MJPEG; a source that cannot be opened is a 503."
    ),
    version=__version__,
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
    return {"service": "ppe-detection", "version": __version__, "docs": "/docs"}


@app.get("/health", response_model=HealthOut, tags=["meta"])
def health() -> HealthOut:
    status = runtime_status()
    return HealthOut(
        status="ok" if status.ready else "not_ready",
        ready=status.ready,
        version=__version__,
        weights=str(status.weights),
        weights_exists=status.weights_exist,
        backend=status.backend,
        execution=status.execution,
        provider=status.provider,
        npu_available=status.npu_available,
        device=status.device,
        detail=status.detail,
    )


@app.get("/devices", tags=["meta"])
def devices() -> dict[str, Any]:
    """Which execution providers this host can bind, and which are NPUs."""
    return describe_environment()


@app.get("/classes", tags=["meta"])
def classes() -> dict[str, Any]:
    return {"count": len(UNIFIED_CLASS_NAMES), "names": UNIFIED_CLASS_NAMES}


@app.get("/metrics", tags=["meta"])
def metrics() -> dict[str, Any]:
    """Latency and tracker state for whatever the process has served so far."""
    try:
        pipeline = get_pipeline()
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return pipeline.stats()


@app.post("/predict/image", response_model=ImagePredictOut, tags=["predict"])
async def predict_image(
    file: UploadFile = File(..., description="JPEG, PNG, or WebP frame"),
    conf: float = Query(0.25, gt=0, lt=1, description="Confidence threshold"),
    return_image: bool = Query(False, description="Include an annotated JPEG as base64"),
) -> ImagePredictOut:
    try:
        image = decode_image_bytes(await file.read())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    pipeline = _pipeline_or_503(conf)
    with PIPELINE_LOCK:
        # A still is not part of a stream, so it must not inherit track ids.
        pipeline.reset()
        result = pipeline.process(image)
        annotated = annotate(image, result.detections, result.workers) if return_image else None

    encoded = (
        base64.b64encode(encode_jpeg(annotated)).decode("ascii") if annotated is not None else None
    )
    workers = [WorkerOut(**worker.as_dict()) for worker in result.workers]
    return ImagePredictOut(
        detections=[
            DetectionOut(cls_name=d.cls_name, conf=d.conf, xyxy=list(d.xyxy))
            for d in result.detections
        ],
        compliance=workers,
        labels=[worker.label for worker in workers],
        latency_ms=round(result.latency_ms, 3),
        annotated_jpeg_base64=encoded,
    )


@app.post("/predict/video", response_model=VideoPredictOut, tags=["predict"])
async def predict_video(
    file: UploadFile = File(..., description="Video file (mp4, avi, mov)"),
    conf: float = Query(0.25, gt=0, lt=1, description="Confidence threshold"),
    return_clip: bool = Query(False, description="Write an annotated mp4 under output/app/"),
    max_frames: int = Query(DEFAULT_MAX_VIDEO_FRAMES, ge=1, le=2000),
    max_seconds: float = Query(DEFAULT_MAX_VIDEO_SECONDS, ge=1, le=300),
) -> VideoPredictOut:
    """Score a capped, strided sample of an uploaded clip and summarise it."""
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    tmp_path: str | None = None
    cap = None
    writer = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        try:
            cap = open_capture(tmp_path)
        except SourceUnavailable as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        info = capture_info(cap, tmp_path)
        stride = frame_stride(info.frame_count, max_frames)
        pipeline = _pipeline_or_503(conf)

        annotated_path = CLIP_DIR / f"{uuid.uuid4().hex}.mp4" if return_clip else None
        counts: Counter[str] = Counter()
        labels: list[str] = []
        seen: set[str] = set()
        events: list[str] = []
        processed = 0
        workers_max = 0
        violation_frames = 0

        with PIPELINE_LOCK:
            pipeline.reset()
            for _index, frame, timestamp in iter_capped_frames(
                cap, max_frames=max_frames, max_seconds=max_seconds
            ):
                result = pipeline.process(frame, timestamp)
                processed += 1
                workers_max = max(workers_max, len(result.workers))
                if any(worker.violations for worker in result.workers):
                    violation_frames += 1
                for worker in result.workers:
                    counts.update(worker.violations)
                    if worker.label not in seen and len(labels) < 50:
                        seen.add(worker.label)
                        labels.append(worker.label)
                events.extend(event.describe() for event in result.events)

                if annotated_path is not None:
                    if writer is None:
                        height, width = frame.shape[:2]
                        writer = open_writer(annotated_path, info.fps / stride, (width, height))
                    writer.write(annotate(frame, result.detections, result.workers))
            latency = pipeline.latency.as_dict()

        if writer is not None:
            writer.release()
            writer = None

        clip_ready = annotated_path is not None and annotated_path.is_file()
        return VideoPredictOut(
            source_fps=info.fps,
            frame_count=info.frame_count,
            frames_processed=processed,
            stride=stride,
            capped=processed < info.frame_count if info.frame_count > 0 else False,
            workers_per_frame_max=workers_max,
            violation_frames=violation_frames,
            violation_counts=dict(counts),
            sample_labels=labels,
            events=events[:50],
            latency={k: float(v) for k, v in latency.items()},
            annotated_path=str(annotated_path) if clip_ready else None,
            annotated_url=f"/clips/{annotated_path.name}" if clip_ready else None,
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
    """Serve a clip written by ``POST /predict/video?return_clip=true``."""
    if Path(name).name != name or not name.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Invalid clip name.")
    path = CLIP_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Clip not found.")
    return FileResponse(path, media_type="video/mp4", filename=name)


@app.get("/stream", tags=["stream"])
def stream(
    source: str | None = Query(None, description="Camera index, RTSP URL, or file path"),
    conf: float = Query(0.25, gt=0, lt=1),
    max_fps: float = Query(12.0, ge=1, le=30, description="Send rate cap"),
) -> StreamingResponse:
    """MJPEG stream with boxes and compliance overlays.

    Unlike the upload endpoints this keeps tracker state, so alerts debounce
    across frames the way they would on a device.
    """
    try:
        cap = open_capture(source)
    except SourceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        pipeline = _pipeline_or_503(conf)
    except HTTPException:
        cap.release()
        raise

    interval = 1.0 / float(max_fps)

    def frames():
        try:
            while True:
                started = time.perf_counter()
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                with PIPELINE_LOCK:
                    result = pipeline.process(frame)
                    annotated = annotate(frame, result.detections, result.workers)
                jpeg = encode_jpeg(annotated, quality=75)
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                leftover = interval - (time.perf_counter() - started)
                if leftover > 0:
                    time.sleep(leftover)
        finally:
            cap.release()

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store", "Pragma": "no-cache"},
    )


def _pipeline_or_503(conf: float):
    try:
        return get_pipeline(conf=conf)
    except (FileNotFoundError, RuntimeError, ImportError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api.main:app",
        host=os.environ.get("PPE_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("PPE_API_PORT", "8000")),
        reload=True,
    )
