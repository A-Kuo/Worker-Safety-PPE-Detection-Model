"""Execution provider registry and policy resolution.

These run on a CPU-only host, which is the interesting case: the point of the
registry is that a machine without an accelerator says so instead of quietly
serving CPU inference.
"""

from __future__ import annotations

import pytest

from ppe.providers import (
    CPU_PROVIDER,
    EXECUTION_POLICIES,
    PROVIDERS,
    NpuUnavailable,
    ProviderNotBound,
    available_npu_providers,
    describe_environment,
    dynamic_axes,
    installed_providers,
    known_providers,
    probe,
    requires_static_shapes,
    resolve,
    spec_for,
    verify_binding,
)

pytest.importorskip("onnxruntime", reason="the registry probes ONNX Runtime")

HAS_NPU = bool(available_npu_providers())
skip_with_npu = pytest.mark.skipif(HAS_NPU, reason="this host has an NPU provider")


def test_every_registry_name_is_a_real_onnxruntime_provider():
    known = set(known_providers())
    unknown = [spec.name for spec in PROVIDERS if spec.name not in known]
    assert unknown == []


def test_registry_covers_the_major_npu_vendors():
    vendors = {spec.vendor for spec in PROVIDERS if spec.is_npu}
    assert {"Qualcomm", "Intel", "AMD", "Apple", "Huawei"} <= vendors


def test_directml_is_not_classified_as_an_npu():
    # DirectML dispatches to whatever device Windows offers, often a GPU.
    assert spec_for("DmlExecutionProvider").kind == "mixed"


def test_probe_marks_cpu_only_hosts_honestly():
    statuses = {status.spec.name: status.available for status in probe()}
    installed = set(installed_providers())
    for name, available in statuses.items():
        assert available == (name in installed)


def test_qnn_backend_path_is_filled_in():
    options = spec_for("QNNExecutionProvider").provider_options()
    assert options["backend_path"].startswith("libQnnHtp") or options["backend_path"].endswith(
        ".dll"
    )
    assert options["htp_performance_mode"] == "burst"


def test_openvino_pins_the_npu_device():
    assert spec_for("OpenVINOExecutionProvider").provider_options()["device_type"] == "NPU"


def test_provider_options_accept_overrides():
    options = spec_for("OpenVINOExecutionProvider").provider_options({"device_type": "GPU"})
    assert options["device_type"] == "GPU"


def test_cpu_policy_resolves_to_cpu():
    assert resolve("cpu") == [(CPU_PROVIDER, {})]


@skip_with_npu
def test_strict_npu_raises_when_nothing_is_installed():
    with pytest.raises(NpuUnavailable) as excinfo:
        resolve("npu")
    message = str(excinfo.value)
    assert "requires an NPU execution provider" in message
    assert "onnxruntime-qnn" in message
    assert "--execution cpu" in message


@skip_with_npu
def test_npu_preferred_degrades_to_cpu():
    assert resolve("npu-preferred") == [(CPU_PROVIDER, {})]


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="Unknown execution policy"):
        resolve("gpu")


def test_policies_are_the_documented_three():
    assert EXECUTION_POLICIES == ("npu", "npu-preferred", "cpu")


def test_pinning_an_unknown_provider_name():
    with pytest.raises(ValueError, match="is not an ONNX Runtime provider"):
        resolve("cpu", provider="HexagonMagicProvider")


@skip_with_npu
def test_pinning_an_uninstalled_npu_provider_explains_the_install():
    with pytest.raises(NpuUnavailable, match="onnxruntime-openvino"):
        resolve("npu", provider="OpenVINOExecutionProvider")


def test_pinning_cpu_is_allowed():
    assert resolve("cpu", provider=CPU_PROVIDER) == [(CPU_PROVIDER, {})]


def test_static_shape_requirements():
    assert requires_static_shapes("QNNExecutionProvider") is True
    assert requires_static_shapes("CoreMLExecutionProvider") is False
    assert requires_static_shapes(CPU_PROVIDER) is False


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ([1, 3, 640, 640], []),
        ([1, 3, "height", "width"], [2, 3]),
        (["batch", 3, 640, 640], [0]),
        ([1, 3, -1, 640], [2]),
    ],
)
def test_dynamic_axes(shape, expected):
    assert dynamic_axes(shape) == expected


def test_verify_binding_accepts_a_match():
    session = _FakeSession(["QNNExecutionProvider", CPU_PROVIDER])
    active = verify_binding(session, [("QNNExecutionProvider", {})], "npu")
    assert active == "QNNExecutionProvider"


def test_verify_binding_catches_a_silent_cpu_fallback():
    session = _FakeSession([CPU_PROVIDER])
    with pytest.raises(ProviderNotBound, match="falls back silently"):
        verify_binding(session, [("QNNExecutionProvider", {})], "npu")


def test_verify_binding_is_permissive_off_the_strict_policy():
    session = _FakeSession([CPU_PROVIDER])
    assert verify_binding(session, [("QNNExecutionProvider", {})], "npu-preferred") == CPU_PROVIDER


def test_describe_environment_is_json_ready():
    env = describe_environment()
    assert env["onnxruntime"]
    assert CPU_PROVIDER in env["installed_providers"]
    assert len(env["providers"]) == len(PROVIDERS)
    row = env["providers"][0]
    assert {"provider", "vendor", "hardware", "available", "install"} <= set(row)


@skip_with_npu
def test_unavailable_rows_carry_an_install_hint():
    for row in describe_environment()["providers"]:
        assert row["install"] if not row["available"] else row["install"] is None


class _FakeSession:
    """Minimal stand-in for an InferenceSession's provider reporting."""

    def __init__(self, providers: list[str]) -> None:
        self._providers = providers

    def get_providers(self) -> list[str]:
        return list(self._providers)
