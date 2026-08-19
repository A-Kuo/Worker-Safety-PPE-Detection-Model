"""Shared fixtures.

The suite runs without a checkpoint, a GPU, or a camera. Anything that needs a
real model builds a tiny synthetic one, and anything that needs frames draws
them with numpy.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import HELMET_BOX, PERSON_BOX, VEST_BOX, detection

from ppe.backends import StubBackend
from ppe.config import RuntimeConfig
from ppe.pipeline import EdgePipeline
from ppe.schema import UNIFIED_CLASS_NAMES


@pytest.fixture
def unified_names() -> dict[int, str]:
    return dict(enumerate(UNIFIED_CLASS_NAMES))


@pytest.fixture
def frame() -> np.ndarray:
    """A 480x640 BGR frame with a little structure, not flat noise."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:, :] = (40, 45, 50)
    image[60:400, 100:220] = (90, 100, 120)
    image[62:110, 130:190] = (30, 200, 240)
    return image


@pytest.fixture
def clock():
    """A manual clock so alert timing is deterministic."""

    class Clock:
        def __init__(self) -> None:
            self.now = 0.0

        def __call__(self) -> float:
            return self.now

        def advance(self, seconds: float) -> float:
            self.now += seconds
            return self.now

    return Clock()


@pytest.fixture
def make_backend():
    def build(frames, names=None, loop=True) -> StubBackend:
        return StubBackend(frames=frames, names=names, loop=loop)

    return build


@pytest.fixture
def make_pipeline(make_backend):
    def build(frames, backend=None, **config_kwargs) -> EdgePipeline:
        config = RuntimeConfig(backend="stub", **config_kwargs)
        return EdgePipeline(backend or make_backend(frames), config)

    return build


@pytest.fixture
def compliant_frame():
    return [
        detection("person", PERSON_BOX),
        detection("helmet", HELMET_BOX),
        detection("vest", VEST_BOX),
    ]


@pytest.fixture
def bare_head_frame():
    return [
        detection("person", PERSON_BOX),
        detection("no_helmet", HELMET_BOX, conf=0.8),
        detection("vest", VEST_BOX),
    ]
