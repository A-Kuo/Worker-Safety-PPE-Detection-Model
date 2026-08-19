"""Runtime configuration and environment parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from ppe.config import RuntimeConfig, config_from_env


def test_defaults_are_usable():
    config = RuntimeConfig()
    assert config.conf == 0.25
    assert config.required_ppe == ("helmet", "vest")
    assert config.imgsz == 640


def test_backend_auto_follows_the_weights_extension():
    assert RuntimeConfig(weights=Path("model.onnx")).backend_name == "onnx"
    assert RuntimeConfig(weights=Path("best.pt")).backend_name == "ultralytics"
    assert RuntimeConfig().backend_name == "ultralytics"


def test_an_explicit_backend_wins():
    config = RuntimeConfig(weights=Path("best.pt"), backend="stub")
    assert config.backend_name == "stub"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"conf": 0.0}, "conf"),
        ({"conf": 1.0}, "conf"),
        ({"iou": 2.0}, "iou"),
        ({"imgsz": 641}, "imgsz"),
        ({"imgsz": 0}, "imgsz"),
        ({"frame_stride": 0}, "frame_stride"),
        ({"alert_min_frames": 0}, "alert_min_frames"),
        ({"required_ppe": ("jetpack",)}, "unknown classes"),
    ],
)
def test_invalid_settings_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RuntimeConfig(**kwargs)


def test_with_overrides_ignores_none():
    config = RuntimeConfig(conf=0.4)
    assert config.with_overrides(conf=None).conf == 0.4
    assert config.with_overrides(conf=0.6).conf == 0.6


def test_config_is_frozen():
    config = RuntimeConfig()
    with pytest.raises(AttributeError):
        config.conf = 0.9


def test_env_parsing():
    config = config_from_env(
        {
            "PPE_WEIGHTS": "models/site.onnx",
            "PPE_CONF": "0.4",
            "PPE_IMGSZ": "320",
            "PPE_REQUIRED_PPE": "helmet, vest, goggles",
            "PPE_ALERT_FRAMES": "5",
        }
    )
    assert config.weights == Path("models/site.onnx")
    assert config.backend_name == "onnx"
    assert config.conf == 0.4
    assert config.imgsz == 320
    assert config.required_ppe == ("helmet", "vest", "goggles")


def test_env_falls_back_to_defaults_when_empty():
    config = config_from_env({"PPE_WEIGHTS": "  ", "PPE_REQUIRED_PPE": ""})
    assert config.weights is None
    assert config.required_ppe == ("helmet", "vest")


def test_env_ignores_unrelated_variables():
    config = config_from_env({"PATH": "/usr/bin", "CONF": "0.9"})
    assert config.conf == 0.25
