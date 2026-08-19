"""Streamlit review UI: upload a frame or clip and read the compliance calls."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st  # noqa: E402

from app.media import (  # noqa: E402
    DEFAULT_MAX_VIDEO_FRAMES,
    DEFAULT_MAX_VIDEO_SECONDS,
    bgr_to_rgb,
    decode_image_bytes,
    iter_capped_frames,
)
from app.paths import ensure_src_on_path, resolve_weights_path  # noqa: E402
from app.runtime import PIPELINE_LOCK, get_pipeline  # noqa: E402

ensure_src_on_path()

from ppe.draw import annotate  # noqa: E402
from ppe.sources import SourceUnavailable, open_capture  # noqa: E402

st.set_page_config(page_title="PPE compliance", layout="wide")


def load_pipeline(weights: str, conf: float):
    return get_pipeline(weights or None, conf=conf)


def show_workers(workers) -> None:
    if not workers:
        st.info("No people detected in this frame.")
        return
    compliant = sum(1 for worker in workers if worker.compliant)
    left, middle, right = st.columns(3)
    left.metric("Workers", len(workers))
    middle.metric("Compliant", compliant)
    right.metric("Violations", len(workers) - compliant)

    for worker in workers:
        details = []
        if worker.present:
            details.append("wearing: " + ", ".join(worker.present))
        if worker.missing:
            details.append("missing: " + ", ".join(worker.missing))
        if worker.violations:
            details.append("flags: " + ", ".join(worker.violations))
        body = worker.label if not details else f"{worker.label}  \n" + " | ".join(details)
        (st.success if worker.compliant else st.error)(body)


def image_tab(pipeline) -> None:
    upload = st.file_uploader(
        "Upload a still (JPEG, PNG, WebP)",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_upload",
    )
    if upload is None:
        return
    if pipeline is None:
        st.info("Set a valid weights path in the sidebar.")
        return
    try:
        image = decode_image_bytes(upload.getvalue())
    except ValueError as exc:
        st.error(str(exc))
        return

    with PIPELINE_LOCK:
        pipeline.reset()
        result = pipeline.process(image)
        annotated = annotate(image, result.detections, result.workers)

    left, right = st.columns((3, 2))
    with left:
        st.image(bgr_to_rgb(annotated), caption="Annotated", use_container_width=True)
    with right:
        st.subheader("Compliance")
        st.caption(f"{result.latency_ms:.1f} ms")
        show_workers(result.workers)


def video_tab(pipeline, max_frames: int) -> None:
    clip = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        key="video_upload",
    )
    if clip is None:
        return
    if pipeline is None:
        st.info("Set a valid weights path in the sidebar.")
        return

    scratch = _REPO_ROOT / "output" / "app" / f"ui_upload{Path(clip.name).suffix or '.mp4'}"
    scratch.parent.mkdir(parents=True, exist_ok=True)
    scratch.write_bytes(clip.getvalue())

    try:
        cap = open_capture(str(scratch))
    except SourceUnavailable as exc:
        st.error(str(exc))
        return

    progress = st.progress(0, text="Running detection")
    last_frame = None
    last_workers: list = []
    alerts: list[str] = []
    processed = 0
    workers_max = 0
    violation_frames = 0

    try:
        with PIPELINE_LOCK:
            pipeline.reset()
            for index, frame, timestamp in iter_capped_frames(
                cap, max_frames=max_frames, max_seconds=DEFAULT_MAX_VIDEO_SECONDS
            ):
                result = pipeline.process(frame, timestamp)
                last_frame = annotate(frame, result.detections, result.workers)
                last_workers = result.workers
                processed += 1
                workers_max = max(workers_max, len(result.workers))
                if any(worker.violations for worker in result.workers):
                    violation_frames += 1
                alerts.extend(event.describe() for event in result.events)
                progress.progress(
                    min(processed / max(max_frames, 1), 1.0),
                    text=f"Frame {index}, {processed} inferred",
                )
            latency = pipeline.latency.as_dict()
    finally:
        cap.release()
        progress.empty()

    one, two, three = st.columns(3)
    one.metric("Frames inferred", processed)
    two.metric("Workers (max)", workers_max)
    three.metric("Frames with violations", violation_frames)
    st.caption(f"Median {latency.get('p50_ms', 0)} ms per frame, {latency.get('fps', 0)} fps")

    if last_frame is not None:
        st.image(bgr_to_rgb(last_frame), caption="Last annotated frame", use_container_width=True)
    if alerts:
        st.subheader("Alerts raised")
        for line in alerts[:40]:
            st.write(line)
    show_workers(last_workers)


def webcam_tab(pipeline) -> None:
    st.write(
        "Single snapshot from the browser camera. For a continuous preview, run the API "
        "and open /stream."
    )
    try:
        snap = st.camera_input("Capture a frame")
    except Exception:
        st.warning("The camera widget is unavailable in this Streamlit build or browser.")
        return
    if snap is None:
        return
    if pipeline is None:
        st.info("Set a valid weights path in the sidebar.")
        return
    try:
        image = decode_image_bytes(snap.getvalue())
    except ValueError as exc:
        st.error(str(exc))
        return
    with PIPELINE_LOCK:
        pipeline.reset()
        result = pipeline.process(image)
        annotated = annotate(image, result.detections, result.workers)
    st.image(bgr_to_rgb(annotated), caption="Annotated", use_container_width=True)
    show_workers(result.workers)


def main() -> None:
    st.title("Worker PPE compliance")
    st.caption('Detections attached to each person, reported as "Worker 12: missing helmet".')

    with st.sidebar:
        st.header("Model")
        weights = st.text_input("Weights path", value=str(resolve_weights_path()))
        conf = st.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.05)
        max_frames = st.slider(
            "Max video frames",
            30,
            600,
            DEFAULT_MAX_VIDEO_FRAMES,
            30,
            help="Long clips are sampled across their duration, not cut off at the start.",
        )
        st.caption(
            "Live MJPEG comes from the API at GET /stream. Start uvicorn and open "
            "http://127.0.0.1:8000/stream. Hosts without a camera get a 503."
        )

    pipeline = None
    try:
        pipeline = load_pipeline(weights.strip(), float(conf))
    except Exception as exc:
        st.warning(f"Could not load the detector: {exc}")

    tabs = st.tabs(["Image", "Video", "Webcam"])
    with tabs[0]:
        image_tab(pipeline)
    with tabs[1]:
        video_tab(pipeline, int(max_frames))
    with tabs[2]:
        webcam_tab(pipeline)


main()
