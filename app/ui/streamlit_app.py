"""Streamlit demo: camera-first PPE review with NPU-oriented edge runtime."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st
import streamlit.components.v1 as components

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
from app.runtime import IMPORT_ERROR, build_engine, runtime_status

ensure_src_on_path()
try:
    from ppe.compliance import WorkerCompliance
except ImportError:
    from app.runtime import WorkerCompliance

st.set_page_config(
    page_title="PPE Compliance · Camera",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Compact industrial look — avoid purple/cream AI defaults.
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Serif:wght@600&display=swap');
      html, body, [class*="css"]  { font-family: 'IBM Plex Sans', sans-serif; }
      .block-container { padding-top: 1.25rem; max-width: 1200px; }
      h1, h2, h3 { font-family: 'IBM Plex Serif', Georgia, serif !important; letter-spacing: -0.02em; }
      div[data-testid="stMetric"] {
        background: linear-gradient(160deg, #f3f6f4 0%, #e4ebe6 100%);
        border: 1px solid #c5d0c8;
        padding: 0.6rem 0.85rem;
        border-radius: 2px;
      }
      .ppe-banner {
        background: linear-gradient(120deg, #1c2b24 0%, #2f4a3c 55%, #3d5c4a 100%);
        color: #f2f7f4;
        padding: 1.1rem 1.35rem;
        border-radius: 2px;
        margin-bottom: 0.75rem;
      }
      .ppe-banner h1 { color: #f2f7f4 !important; margin: 0 0 0.25rem 0; font-size: 1.85rem; }
      .ppe-banner p { margin: 0; opacity: 0.9; font-size: 0.95rem; }
      .ppe-chip {
        display: inline-block; margin-right: 0.4rem; margin-top: 0.55rem;
        padding: 0.15rem 0.55rem; border: 1px solid rgba(242,247,244,0.35);
        font-size: 0.75rem; letter-spacing: 0.04em; text-transform: uppercase;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def _cached_engine(
    weights: str,
    device: str,
    prefer_npu: bool,
    npu_only: bool,
    openvino_device: str,
    providers: str,
) -> object:
    provider_tuple = tuple(p.strip() for p in providers.split(",") if p.strip()) or None
    return build_engine(
        weights,
        conf=0.25,
        device=device or None,
        prefer_npu=prefer_npu,
        npu_only=npu_only if npu_only else None,
        openvino_device=openvino_device or None,
        providers=provider_tuple,
    )


def _load_engine(
    weights: str,
    conf: float,
    device: str,
    prefer_npu: bool,
    npu_only: bool,
    openvino_device: str,
    providers: str,
):
    engine = _cached_engine(weights, device, prefer_npu, npu_only, openvino_device, providers)
    engine.conf = float(conf)
    return engine


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


def _run_frame(engine, image_bgr):
    timed = engine.predict_timed(image_bgr)
    return timed.annotated, timed.workers, timed.latency_ms


def _provider_summary(status: dict) -> str:
    resolved = status.get("resolved_providers") or []
    if not resolved:
        err = status.get("error")
        return f"Providers unavailable{': ' + err if err else ''}"
    return " → ".join(resolved)


def main() -> None:
    st.markdown(
        """
        <div class="ppe-banner">
          <h1>Worker PPE compliance</h1>
          <p>Camera-first review on the NPU edge path — ONNX Runtime prefers NPU, then GPU, then CPU.</p>
          <span class="ppe-chip">Live camera</span>
          <span class="ppe-chip">Timely frames</span>
          <span class="ppe-chip">NPU first</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if IMPORT_ERROR:
        st.error(
            "The `ppe` package is not importable yet. "
            f"Prefer `pip install -e .` or put `src/` on PYTHONPATH. ({IMPORT_ERROR})"
        )
        st.stop()

    default_weights = str(resolve_weights_path())
    status = runtime_status(prefer_npu=True)

    with st.sidebar:
        st.header("Runtime")
        st.caption(_provider_summary(status))
        if status.get("openvino_device_type"):
            st.caption(f"OpenVINO device: `{status['openvino_device_type']}`")

        weights = st.text_input("Weights path", value=default_weights)
        conf = st.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.05)
        prefer_npu = st.toggle(
            "Prefer NPU",
            value=True,
            help="Select NPU execution providers first (OpenVINO / QNN / CoreML / CANN).",
        )
        npu_only = st.toggle(
            "NPU only (CPU soft-fallback still on unless env disables it)",
            value=bool(os.environ.get("PPE_NPU_ONLY", "").lower() in {"1", "true", "yes", "on"}),
            help="Skip CUDA/TensorRT. Matches PPE_NPU_ONLY.",
        )
        openvino_device = st.text_input(
            "OpenVINO device_type",
            value=os.environ.get("PPE_OPENVINO_DEVICE") or "AUTO:NPU,GPU",
            help="e.g. AUTO:NPU,GPU · NPU · GPU · HETERO:NPU,GPU,CPU",
        )
        providers = st.text_input(
            "Providers pin (optional)",
            value=os.environ.get("PPE_PROVIDERS", ""),
            help="Comma keys: openvino,qnn,coreml,cann,cuda,cpu",
        )
        device = st.text_input(
            "Torch device (blank = auto / PPE_DEVICE)",
            value="",
            help="Only used when falling back to .pt weights.",
        )
        max_frames = st.slider(
            "Max video frames",
            30,
            600,
            DEFAULT_MAX_VIDEO_FRAMES,
            30,
        )
        api_base = st.text_input(
            "API base URL (live MJPEG)",
            value=os.environ.get("PPE_API_URL", "http://127.0.0.1:8000"),
            help="Start uvicorn for continuous camera: uvicorn app.api.main:app --port 8000",
        )
        stream_source = st.text_input("Stream source", value="0", help="Webcam index, RTSP, or file path")
        stream_fps = st.slider("Stream max FPS", 1, 30, 12)

    engine = None
    load_error = None
    try:
        engine = _load_engine(
            weights.strip(),
            float(conf),
            device.strip(),
            prefer_npu,
            npu_only,
            openvino_device.strip(),
            providers.strip(),
        )
    except FileNotFoundError as exc:
        load_error = str(exc)
    except Exception as exc:
        load_error = f"Could not load engine: {exc}"
    if load_error:
        st.warning(load_error)

    info = engine.info() if engine is not None else {}
    backend_info = info.get("backend") if isinstance(info.get("backend"), dict) else {}
    active = backend_info.get("providers_active") or status.get("resolved_providers") or []
    active_label = (active[0] if active else "—").replace("ExecutionProvider", "")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Engine", str(info.get("engine") or "—"))
    m2.metric("Active EP", active_label)
    m3.metric("OpenVINO", backend_info.get("openvino_device_type") or status.get("openvino_device_type") or "—")
    m4.metric("Model", Path(str(info.get("model_path") or weights)).name)

    cam_tab, image_tab, video_tab = st.tabs(["Camera", "Image", "Video"])

    with cam_tab:
        st.subheader("Live camera")
        st.caption(
            "Continuous MJPEG uses the API with timely frame draining so the NPU "
            "always works on the newest frame. Snapshots run in-process below."
        )
        live = st.toggle("Show live API stream", value=True)
        if live:
            stream_url = (
                f"{api_base.rstrip('/')}/stream"
                f"?source={stream_source.strip() or '0'}"
                f"&conf={conf}&max_fps={stream_fps}&timely=true"
            )
            st.caption(stream_url)
            components.html(
                f"""
                <div style="background:#111814;padding:8px;border-radius:2px;">
                  <img src="{stream_url}"
                       style="width:100%;max-height:520px;object-fit:contain;background:#0b100d;"
                       alt="Live PPE stream (503 if no camera / API down)" />
                </div>
                """,
                height=540,
            )
            st.caption("If the image is blank, start the API and attach a camera (HTTP 503 otherwise).")

        st.divider()
        st.subheader("Snapshot")
        snap = None
        try:
            snap = st.camera_input("Capture a frame for a timed NPU/CPU pass")
        except Exception:
            st.warning("Webcam widget is unavailable in this Streamlit build or browser.")
        if snap is not None and engine is None:
            st.info("Set a valid weights path in the sidebar.")
        elif snap is not None:
            try:
                image = decode_image_bytes(snap.getvalue())
            except ValueError as exc:
                st.error(str(exc))
            else:
                annotated, workers, latency_ms = _run_frame(engine, image)
                left, right = st.columns((3, 2))
                with left:
                    st.image(
                        bgr_to_rgb(annotated),
                        caption=f"Annotated · {latency_ms:.0f} ms",
                        use_container_width=True,
                    )
                with right:
                    st.metric("Inference", f"{latency_ms:.0f} ms")
                    st.metric("Engine", engine.kind)
                    _show_workers(workers)

    with image_tab:
        upload = st.file_uploader(
            "Upload a still (JPEG, PNG, WebP)",
            type=["jpg", "jpeg", "png", "webp"],
            key="image_upload",
        )
        if upload is not None and engine is None:
            st.info("Set a valid weights path in the sidebar.")
        elif upload is not None:
            try:
                image = decode_image_bytes(upload.getvalue())
            except ValueError as exc:
                st.error(str(exc))
            else:
                annotated, workers, latency_ms = _run_frame(engine, image)
                left, right = st.columns((3, 2))
                with left:
                    st.image(
                        bgr_to_rgb(annotated),
                        caption=f"Annotated · {latency_ms:.0f} ms",
                        use_container_width=True,
                    )
                with right:
                    st.metric("Inference", f"{latency_ms:.0f} ms")
                    st.subheader("Compliance")
                    _show_workers(workers)

    with video_tab:
        clip = st.file_uploader(
            "Upload a video",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            key="video_upload",
        )
        if clip is not None and engine is None:
            st.info("Set a valid weights path in the sidebar.")
        elif clip is not None:
            suffix = Path(clip.name).suffix or ".mp4"
            tmp_path = _REPO_ROOT / "output" / "app" / f"ui_upload{suffix}"
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_bytes(clip.getvalue())

            cap = open_capture(str(tmp_path), low_latency=False)
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
                latencies: list[float] = []
                labels: list[str] = []
                seen: set[str] = set()
                try:
                    for index, frame in iter_capped_frames(
                        cap,
                        max_frames=int(max_frames),
                        max_seconds=DEFAULT_MAX_VIDEO_SECONDS,
                    ):
                        t0 = time.perf_counter()
                        annotated, workers = engine.predict_and_comply(frame)
                        latencies.append((time.perf_counter() - t0) * 1000.0)
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
                            text=f"Frame {index} · {processed} inferred · {latencies[-1]:.0f} ms",
                        )
                finally:
                    cap.release()
                    progress.empty()

                mean_ms = sum(latencies) / len(latencies) if latencies else 0.0
                st.caption(
                    f"Source ~{meta['fps']:.1f} FPS, {int(meta['frame_count'])} frames. "
                    f"Inferred {processed} frames (capped at {max_frames} / "
                    f"{DEFAULT_MAX_VIDEO_SECONDS:.0f}s) · mean {mean_ms:.0f} ms."
                )
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Frames inferred", processed)
                m2.metric("Workers (max)", workers_max)
                m3.metric("Frames with violations", violation_frames)
                m4.metric("Mean latency", f"{mean_ms:.0f} ms")
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


main()
