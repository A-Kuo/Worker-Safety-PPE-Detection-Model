"""Single-shot detector facade.

:class:`EdgePipeline` is the right entry point for streams, where tracking and
alert debouncing matter. This wrapper is for the one-image case: load weights,
hand it a frame, get boxes and compliance back.
"""

from __future__ import annotations

from pathlib import Path

from ppe.backends import DetectorBackend, load_backend
from ppe.compliance import Detection, WorkerCompliance
from ppe.config import RuntimeConfig
from ppe.draw import annotate
from ppe.pipeline import EdgePipeline, FrameResult


class PPEDetector:
    """Load a checkpoint and score frames one at a time."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        conf: float = 0.25,
        device: str | None = None,
        backend: DetectorBackend | None = None,
        config: RuntimeConfig | None = None,
    ) -> None:
        if config is None:
            config = RuntimeConfig(
                weights=Path(weights_path) if weights_path else None,
                conf=conf,
                device=device,
            )
        self.config = config
        self.weights_path = str(config.weights) if config.weights else ""
        self._pipeline = EdgePipeline(backend or load_backend(config), config)

    @property
    def conf(self) -> float:
        return self._pipeline.config.conf

    @conf.setter
    def conf(self, value: float) -> None:
        value = float(value)
        self._pipeline.config = self._pipeline.config.with_overrides(conf=value)
        backend = self._pipeline.backend
        if hasattr(backend, "conf"):
            backend.conf = value

    def names(self) -> dict[int, str]:
        return self._pipeline.backend.class_names()

    def predict_image(self, image) -> list[Detection]:
        return self._run(image).detections

    def predict_and_comply(self, image) -> tuple[object, list[WorkerCompliance]]:
        """Return an annotated BGR frame alongside the compliance records."""
        result = self._run(image)
        return annotate(image, result.detections, result.workers), result.workers

    def close(self) -> None:
        self._pipeline.close()

    def _run(self, image) -> FrameResult:
        # Stateless per call: a still image should not inherit track ids or
        # alert streaks from whatever was scored before it.
        self._pipeline.reset()
        return self._pipeline.process(image)
