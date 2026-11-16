"""PySide6 live window: camera/video in, annotated frames + an honest HUD out.

Qt is used (not ``cv2.imshow``) because the package depends on ``opencv-python-headless``, which has
no GUI, and installing the non-headless wheel next to it corrupts ``cv2``.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from ppe import about
from ppe.window.pipeline import Pipeline, device_label, draw_hud, hud_lines

HELP = "Space pause | R record | +/- threshold | A about | Q / Esc quit"


def require_qt():
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
    except ImportError as exc:
        raise RuntimeError(
            "The live window needs PySide6, which the runtime install does not include. "
            "Install it with: pip install 'ppe[gui]'"
        ) from exc
    return QtCore, QtGui, QtWidgets


def verify_in_background(session: Any, holder: dict) -> threading.Thread | None:
    """Profile node placement once, off the UI thread, so the HUD can say *verified* (or not)."""
    backend = session.backend
    if getattr(backend, "name", "") != "onnxruntime":
        return None

    def run() -> None:
        try:
            from ppe.runtime.placement import verify_placement

            holder["placement"] = verify_placement(session.model_path, session.policy, runs=1)
        except Exception as exc:  # noqa: BLE001 - the HUD reports it; the stream keeps running
            holder["placement"] = {"ok": None, "error": str(exc)}

    thread = threading.Thread(target=run, name="ppe-verify", daemon=True)
    thread.start()
    return thread


def run_window(
    session: Any,
    source: Any,
    *,
    selftest_frames: int = 0,
    record: Path | None = None,
    verify: bool = True,
    out_dir: Path | None = None,
    snapshot: Path | None = None,
) -> dict[str, Any]:
    """Open the window and run until closed (or until ``selftest_frames`` frames were displayed).

    Returns a summary dict (frames shown, FPS, latency, device, placement).
    """
    QtCore, QtGui, QtWidgets = require_qt()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    selftest = selftest_frames > 0
    # Self-test measures the pipeline, so it processes every frame unpaced; live use drops stale frames.
    pipeline = Pipeline(session, source, lossless=True if selftest else None, pace=not selftest, record=record)
    info = session.info()
    holder: dict[str, Any] = {}
    verifier = verify_in_background(session, holder) if verify else None
    out_dir = Path(out_dir or Path.cwd())

    class Window(QtWidgets.QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(f"{about.NAME} - {source.label}")
            self.view = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
            self.view.setMinimumSize(320, 240)
            self.view.setStyleSheet("background:#111;")
            self.setCentralWidget(self.view)
            self.statusBar().showMessage(HELP)
            self.resize(960, 720)
            self.shown = 0
            self.last_counts: dict[str, int] = {}
            self.last_size: tuple[int, int] | None = None
            self.error: str | None = None
            self.first_shown_at = 0.0
            self.last_shown_at = 0.0
            self.last_image = None
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.tick)
            self.timer.start(5)

        # -- frame loop ----------------------------------------------------------------------
        def tick(self) -> None:
            if pipeline.error is not None:
                self.fail(f"{type(pipeline.error).__name__}: {pipeline.error}")
                return
            result = pipeline.results.take(timeout=0)
            if result is None:
                if pipeline.finished.is_set():
                    self.close()
                return
            self.last_counts = result.counts
            image = result.image
            self.last_size = (image.shape[1], image.shape[0])
            lines = hud_lines(
                pipeline.stats.snapshot(),
                result.counts,
                device=device_label(info, holder.get("placement")),
                conf=session.policy.conf,
                paused=pipeline.paused.is_set(),
                recording=pipeline.recording_path is not None,
            )
            draw_hud(image, lines)
            h, w = image.shape[:2]
            qimg = QtGui.QImage(image.data, w, h, 3 * w, QtGui.QImage.Format_BGR888)
            pix = QtGui.QPixmap.fromImage(qimg)  # copies: `image` may be freed after this call
            self.view.setPixmap(
                pix.scaled(self.view.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            )
            self.shown += 1
            now = time.perf_counter()
            self.first_shown_at = self.first_shown_at or now
            self.last_shown_at = now
            self.last_image = image
            if selftest and self.shown >= selftest_frames:
                self.close()

        def fail(self, message: str) -> None:
            self.error = message
            self.timer.stop()
            if not selftest:
                QtWidgets.QMessageBox.critical(self, about.NAME, message)
            self.close()

        # -- controls ------------------------------------------------------------------------
        def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API name
            key = event.key()
            Qt = QtCore.Qt
            if key in (Qt.Key_Q, Qt.Key_Escape):
                self.close()
            elif key == Qt.Key_Space:
                self.statusBar().showMessage("Paused" if pipeline.toggle_pause() else HELP)
            elif key in (Qt.Key_Plus, Qt.Key_Equal):
                self.statusBar().showMessage(f"Threshold {pipeline.adjust_conf(+0.05):.2f}")
            elif key in (Qt.Key_Minus, Qt.Key_Underscore):
                self.statusBar().showMessage(f"Threshold {pipeline.adjust_conf(-0.05):.2f}")
            elif key == Qt.Key_R:
                self.toggle_record()
            elif key == Qt.Key_A:
                self.show_about()
            else:
                super().keyPressEvent(event)

        def toggle_record(self) -> None:
            if pipeline.recording_path is not None:
                self.statusBar().showMessage(f"Saved {pipeline.stop_recording()}")
            elif self.last_size:
                path = out_dir / time.strftime("ppe_clip_%Y%m%d_%H%M%S.mp4")
                try:
                    pipeline.start_recording(path, self.last_size)
                    self.statusBar().showMessage(f"Recording to {path}")
                except RuntimeError as exc:
                    self.statusBar().showMessage(str(exc))

        def show_about(self) -> None:
            model = info.get("backend", {}).get("fingerprint", {}).get("model", {}) or {}
            sha = (model.get("sha256") or "unknown")[:16]
            QtWidgets.QMessageBox.about(
                self,
                f"About {about.NAME}",
                f"<b>{about.NAME}</b><br>Licence: {about.LICENSE}<br>"
                f"Source: <a href='{about.SOURCE_URL}'>{about.SOURCE_URL}</a><br><br>"
                f"Model: {Path(str(session.model_path)).name} (sha256 {sha}...)<br>"
                f"Running on: {device_label(info, holder.get('placement'))}<br><br>"
                "Beta. A screening aid, not a safety system.",
            )

        def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
            self.timer.stop()
            super().closeEvent(event)

    window = Window()
    window.show()
    started = time.perf_counter()
    pipeline.start()
    try:
        app.exec()
    finally:
        elapsed = time.perf_counter() - started
        recorded = pipeline.recording_path
        pipeline.stop()
    if verifier is not None:
        verifier.join(timeout=30 if selftest else 0.1)
    snap = pipeline.stats.snapshot()
    placement = holder.get("placement")
    if snapshot is not None and window.last_image is not None:
        import cv2

        Path(snapshot).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(snapshot), window.last_image)
    # Steady-state rate between the first and last displayed frame: excludes window creation and the
    # model's first-inference compile, which dominate a short run.
    span = window.last_shown_at - window.first_shown_at
    display_fps = round((window.shown - 1) / span, 1) if window.shown > 1 and span > 0 else 0.0
    return {
        "ok": window.error is None and pipeline.error is None and window.shown > 0,
        "error": window.error or (str(pipeline.error) if pipeline.error else None),
        "source": source.label,
        "frames_shown": window.shown,
        "frames_processed": snap["processed"],
        "frames_dropped": snap["dropped"],
        "elapsed_s": round(elapsed, 2),
        "display_fps": display_fps,
        "latency_ms": snap["latency_ms"],
        "stages_p50_ms": snap["stages_p50_ms"],
        "device": device_label(info, placement),
        "strict_npu": bool(getattr(session.policy, "strict_npu", False)),
        "placement_ok": None if placement is None else placement.get("ok"),
        "last_counts": window.last_counts,
        "recorded": str(recorded) if recorded else None,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2, default=str))
