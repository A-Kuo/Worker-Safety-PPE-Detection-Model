"""Tests for the vendor-agnostic ORT provider registry and policy."""

from __future__ import annotations

from ppe.runtime.config import ExecutionPolicy, pin_providers
from ppe.runtime.providers import (
    PROVIDER_REGISTRY,
    get_provider,
    list_providers,
    resolve_providers,
)


def test_registry_has_npu_and_cpu():
    keys = {p.key for p in PROVIDER_REGISTRY}
    assert "cpu" in keys
    assert "qnn" in keys
    assert "openvino" in keys
    assert "cann" in keys
    assert "cuda" in keys


def test_get_provider():
    cpu = get_provider("cpu")
    assert cpu.ort_name == "CPUExecutionProvider"
    assert cpu.tier == "cpu"


def test_list_npu_tier():
    npus = list_providers(tier="npu")
    assert all(p.tier == "npu" for p in npus)
    assert {p.key for p in npus} >= {"qnn", "openvino", "cann", "coreml"}


def test_resolve_always_includes_cpu_when_installed(monkeypatch):
    monkeypatch.setattr(
        "ppe.runtime.providers.available_ort_names",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    providers = resolve_providers(None)
    assert providers[0] == "CUDAExecutionProvider"
    assert providers[-1] == "CPUExecutionProvider"


def test_resolve_preferred_pin(monkeypatch):
    monkeypatch.setattr(
        "ppe.runtime.providers.available_ort_names",
        lambda: ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
    )
    providers = resolve_providers(["openvino", "cpu"])
    assert providers[0] == "OpenVINOExecutionProvider"
    assert "CPUExecutionProvider" in providers


def test_npu_only_skips_cuda(monkeypatch):
    monkeypatch.setattr(
        "ppe.runtime.providers.available_ort_names",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    providers = resolve_providers(None, npu_only=True)
    assert "CUDAExecutionProvider" not in providers
    assert providers == ["CPUExecutionProvider"]


def test_policy_from_env(monkeypatch):
    monkeypatch.setenv("PPE_ALLOW_TORCH", "1")
    monkeypatch.setenv("PPE_NPU_ONLY", "true")
    monkeypatch.setenv("PPE_PROVIDERS", "qnn,cpu")
    policy = ExecutionPolicy.from_env()
    assert policy.allow_torch is True
    assert policy.npu_only is True
    assert policy.providers == ("qnn", "cpu")


def test_torch_gated_by_default():
    policy = ExecutionPolicy()
    assert policy.allow_torch is False
    assert policy.prefer_onnx is True


def test_pin_providers():
    assert pin_providers([" openvino ", "", "cpu"]) == ("openvino", "cpu")
