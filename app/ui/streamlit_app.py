"""Streamlit demo: upload a frame or clip, review boxes and compliance strings."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from app.media import (
    DEFAULT_MAX_VIDEO_FRAMES,
    DEFAULT_MAX_VIDEO_SECONDS,
    bgr_to_rgb,
    decode_image_bytes,
    iter_capped_frames,
    open_capture,
    video_meta,
)
from app.paths import ensure_src_on_path, resolve_weights_path
from app.runtime import IMPORT_ERROR, build_detector

ensure_src_on_path()
try:
    from ppe.compliance import Detection, WorkerCompliance, associate_ppe_to_persons
    from ppe.inference import PPEDetector
except ImportError:
    from app.runtime import Detection, PPEDetector, WorkerCompliance, associate_ppe_to_persons

st.set_page_config(page_title="PPE Compliance Demo", layout="wide")


@st.cache_resource
def _cached_detector(weights: str, device: str) -> PPEDetector:
    return build_detector(weights, conf=0.25, device=device or None)


def _load_detector(weights: str, conf: float, device: str) -> PPEDetector:
    detector = _cached_detector(weights, device)
    detector.conf = float(conf)
    return detector


def _show_workers(workers: list[WorkerCompliance]) -> None:
    if not workers:
        st.info("No person boxes in this frame.")
        return
    ok = sum(1 for w in workers if not w.missing and not w.violations)
    bad = len(workers) - ok
    c1, c2, c3 = st.columns(3)
    c1.metric("Workers", len(workers))
    c2.metric("Compliant", ok)
    c3.metric("Violations", bad)
    for worker in workers:
        text = worker.label
        extras = []
        if worker.present:
            extras.append("present: " + ", ".join(worker.present))
        if worker.missing:
            extras.append("missing: " + ", ".join(worker.missing))
        if worker.violations:
            extras.append("flags: " + ", ".join(worker.violations))
        body = text if not extras else f"{text}  \n" + " · ".join(extras)
        if worker.missing or worker.violations:
            st.error(body)
        else:
            st.success(body)


def _run_frame(detector: PPEDetector, image_bgr):
    annotated, workers = detector.predict_and_comply(image_bgr)
    return annotated, workers


def main() -> None:
    st.title("Worker PPE compliance")
    st.caption(
        "YOLOv8 detections associated to each person. "
        "Labels look like “Worker 12 — missing helmet and vest”."
    )

    if IMPORT_ERROR:
        st.error(
            "The `ppe` package is not importable yet. "
            f"Prefer `pip install -e .` or put `src/` on PYTHONPATH. ({IMPORT_ERROR})"
        )
        st.stop()

    default_weights = str(resolve_weights_path())
    with st.sidebar:
        st.header("Model")
        weights = st.text_input("Weights path", value=default_weights)
        conf = st.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.05)
        device = st.text_input(
            "Device (blank = auto / PPE_DEVICE)",
            value="",
            help="Examples: cpu, cuda, 0",
        )
        max_frames = st.slider(
            "Max video frames",
            30,
            600,
            DEFAULT_MAX_VIDEO_FRAMES,
            30,
            help="Long clips are sampled across the duration, not truncated from the start only.",
        )
        st.caption(
            "Live MJPEG (webcam / RTSP) is served by the API at "
            "`GET /stream`. Open http://127.0.0.1:8000/stream after starting uvicorn. "
            "A missing camera returns HTTP 503."
        )

    detector = None
    load_error = None
    try:
        detector = _load_detector(weights.strip(), float(conf), device.strip())
    except FileNotFoundError as exc:
        load_error = str(exc)
    except Exception as exc:
        load_error = f"Could not load detector: {exc}"
    if load_error:
        st.warning(load_error)

    image_tab, video_tab, cam_tab = st.tabs(["Image", "Video", "Webcam"])

    with image_tab:
        upload = st.file_uploader(
            "Upload a still (JPEG, PNG, WebP)",
            type=["jpg", "jpeg", "png", "webp"],
            key="image_upload",
        )
        if upload is not None and detector is None:
            st.info("Set a valid weights path in the sidebar.")
        elif upload is not None:
            try:
                image = decode_image_bytes(upload.getvalue())
            except ValueError as exc:
                st.error(str(exc))
            else:
                annotated, workers = _run_frame(detector, image)
                left, right = st.columns((3, 2))
                with left:
                    st.image(bgr_to_rgb(annotated), caption="Annotated", use_container_width=True)
                with right:
                    st.subheader("Compliance")
                    _show_workers(workers)

    with video_tab:
        clip = st.file_uploader(
            "Upload a video",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            key="video_upload",
        )
        if clip is not None and detector is None:
            st.info("Set a valid weights path in the sidebar.")
        elif clip is not None:
            suffix = Path(clip.name).suffix or ".mp4"
            tmp_path = _REPO_ROOT / "output" / "app" / f"ui_upload{suffix}"
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_bytes(clip.getvalue())

            cap = open_capture(str(tmp_path))
            if not cap.isOpened():
                st.error("Could not open the uploaded video.")
            else:
                meta = video_meta(cap)
                progress = st.progress(0, text="Running detection…")
                last_annotated = None
                last_workers = []
                violation_frames = 0
                processed = 0
                workers_max = 0
                labels: list[str] = []
                seen: set[str] = set()
                try:
                    for index, frame in iter_capped_frames(
                        cap,
                        max_frames=int(max_frames),
                        max_seconds=DEFAULT_MAX_VIDEO_SECONDS,
                    ):
                        annotated, workers = detector.predict_and_comply(frame)
                        last_annotated = annotated
                        last_workers = workers
                        processed += 1
                        workers_max = max(workers_max, len(workers))
                        if any(w.missing or w.violations for w in workers):
                            violation_frames += 1
                        for worker in workers:
                            if worker.label not in seen and len(labels) < 40:
                                seen.add(worker.label)
                                labels.append(worker.label)
                        progress.progress(
                            min(processed / max(int(max_frames), 1), 1.0),
                            text=f"Frame {index} · {processed} inferred",
                        )
                finally:
                    cap.release()
                    progress.empty()

                st.caption(
                    f"Source ~{meta['fps']:.1f} FPS, {int(meta['frame_count'])} frames. "
                    f"Inferred {processed} frames (capped at {max_frames} / "
                    f"{DEFAULT_MAX_VIDEO_SECONDS:.0f}s)."
                )
                m1, m2, m3 = st.columns(3)
                m1.metric("Frames inferred", processed)
                m2.metric("Workers (max)", workers_max)
                m3.metric("Frames with violations", violation_frames)
                if last_annotated is not None:
                    st.image(
                        bgr_to_rgb(last_annotated),
                        caption="Last annotated frame",
                        use_container_width=True,
                    )
                if labels:
                    st.subheader("Sample compliance strings")
                    for line in labels:
                        st.write(line)
                _show_workers(last_workers)

    with cam_tab:
        st.write(
            "Snapshot from the browser camera. For a continuous MJPEG preview, "
            "run the API and open `/stream` (503 if no camera is attached)."
        )
        snap = None
        try:
            snap = st.camera_input("Capture a frame")
        except Exception:
            st.warning("Webcam widget is unavailable in this Streamlit build or browser.")
        if snap is not None and detector is None:
            st.info("Set a valid weights path in the sidebar.")
        elif snap is not None:
            try:
                image = decode_image_bytes(snap.getvalue())
            except ValueError as exc:
                st.error(str(exc))
            else:
                annotated, workers = _run_frame(detector, image)
                st.image(bgr_to_rgb(annotated), caption="Annotated", use_container_width=True)
                _show_workers(workers)


main()
