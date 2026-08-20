"""NPU execution providers for ONNX Runtime.

Every vendor ships its own accelerator and its own execution provider, and
ONNX Runtime will quietly run on the CPU when the one you wanted is missing.
That silence is the problem this module exists to remove: a deployment that
asked for an NPU and got a CPU should fail at startup with the reason, not
serve frames at a tenth of the expected rate and let someone discover it in
production.

The registry below is ordered by preference. Nothing here imports a vendor SDK;
availability is probed through ONNX Runtime itself, so the code is inspectable
on a laptop with no accelerator at all.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

CPU_PROVIDER = "CPUExecutionProvider"

EXECUTION_POLICIES = ("npu", "npu-preferred", "cpu")


@dataclass(frozen=True)
class ProviderSpec:
    """One execution provider and what it needs to work."""

    name: str
    vendor: str
    hardware: str
    kind: str
    install: str
    notes: str
    static_shapes_required: bool = False
    options: Mapping[str, str] = field(default_factory=dict)

    @property
    def is_npu(self) -> bool:
        return self.kind == "npu"

    def provider_options(self, overrides: Mapping[str, str] | None = None) -> dict[str, str]:
        merged = dict(self.options)
        if self.name == "QNNExecutionProvider":
            merged["backend_path"] = _qnn_backend_path()
        merged.update(overrides or {})
        return merged


# Ordered by preference. Dedicated NPUs first, then the runtimes that may or may
# not land on one depending on how the host schedules the work.
PROVIDERS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        name="QNNExecutionProvider",
        vendor="Qualcomm",
        hardware="Hexagon NPU (Snapdragon X, 8cx)",
        kind="npu",
        install="pip install onnxruntime-qnn  (Windows on ARM; needs the QNN SDK)",
        notes="Wants INT8 or INT16 QDQ models. Float32 graphs fall back to CPU node by node.",
        static_shapes_required=True,
        options={"htp_performance_mode": "burst", "profiling_level": "off"},
    ),
    ProviderSpec(
        name="OpenVINOExecutionProvider",
        vendor="Intel",
        hardware="AI Boost NPU (Core Ultra)",
        kind="npu",
        install="pip install onnxruntime-openvino openvino",
        notes="device_type=NPU pins the accelerator. Without it OpenVINO may pick the iGPU.",
        static_shapes_required=True,
        options={"device_type": "NPU", "precision": "FP16"},
    ),
    ProviderSpec(
        name="VitisAIExecutionProvider",
        vendor="AMD",
        hardware="XDNA NPU (Ryzen AI)",
        kind="npu",
        install="pip install onnxruntime-vitisai  (needs the Ryzen AI SDK)",
        notes="Expects an INT8 model and a vaip config file supplied through provider options.",
        static_shapes_required=True,
    ),
    ProviderSpec(
        name="CoreMLExecutionProvider",
        vendor="Apple",
        hardware="Neural Engine",
        kind="npu",
        install="pip install onnxruntime  (macOS builds ship CoreML)",
        notes="MLComputeUnits=CPUAndNeuralEngine keeps work off the GPU. macOS only.",
        options={"MLComputeUnits": "CPUAndNeuralEngine", "ModelFormat": "MLProgram"},
    ),
    ProviderSpec(
        name="CANNExecutionProvider",
        vendor="Huawei",
        hardware="Ascend NPU (310P, 910)",
        kind="npu",
        install="Build onnxruntime with --use_cann against the CANN toolkit",
        notes=(
            "Needs the CANN toolkit and the davinci devices mapped into the container. "
            "device_id selects which Ascend chip when the host has several."
        ),
        static_shapes_required=True,
        options={"device_id": "0"},
    ),
    ProviderSpec(
        name="NnapiExecutionProvider",
        vendor="Android",
        hardware="Vendor NPU via NNAPI",
        kind="npu",
        install="Build onnxruntime with --use_nnapi, or use the Android AAR",
        notes="NNAPI decides the backend itself, so the accelerator is a request, not a guarantee.",
        static_shapes_required=True,
    ),
    ProviderSpec(
        name="RknpuExecutionProvider",
        vendor="Rockchip",
        hardware="RKNN NPU (RK3588 and family)",
        kind="npu",
        install="Build onnxruntime with --use_rknpu on the target board",
        notes="Most Rockchip deployments convert to .rknn instead and skip ONNX Runtime.",
        static_shapes_required=True,
    ),
    ProviderSpec(
        name="VSINPUExecutionProvider",
        vendor="VeriSilicon",
        hardware="Vivante NPU",
        kind="npu",
        install="Build onnxruntime with --use_vsinpu",
        notes="Found on NXP i.MX and similar SoCs.",
        static_shapes_required=True,
    ),
    ProviderSpec(
        name="DmlExecutionProvider",
        vendor="Microsoft",
        hardware="DirectML device (NPU or GPU)",
        kind="mixed",
        install="pip install onnxruntime-directml",
        notes=(
            "DirectML picks whichever device Windows offers, which is often the GPU. "
            "Strict NPU mode excludes it; pin it explicitly if you know the host has "
            "no discrete GPU."
        ),
    ),
)

_BY_NAME = {spec.name: spec for spec in PROVIDERS}


@dataclass(frozen=True)
class ProviderStatus:
    """A registry entry paired with whether this build can actually load it."""

    spec: ProviderSpec
    available: bool

    def as_dict(self) -> dict:
        return {
            "provider": self.spec.name,
            "vendor": self.spec.vendor,
            "hardware": self.spec.hardware,
            "kind": self.spec.kind,
            "available": self.available,
            "static_shapes_required": self.spec.static_shapes_required,
            "notes": self.spec.notes,
            "install": None if self.available else self.spec.install,
        }


class NpuUnavailable(RuntimeError):
    """Raised when strict NPU execution was asked for and no NPU provider loaded."""

    def __init__(self, policy: str, statuses: Sequence[ProviderStatus]) -> None:
        self.policy = policy
        self.statuses = list(statuses)
        missing = [s.spec for s in statuses if not s.available and s.spec.is_npu]
        lines = [
            f"Execution policy {policy!r} requires an NPU execution provider, "
            f"and this ONNX Runtime build has none.",
            f"Installed: {', '.join(installed_providers()) or 'none'}",
            "Install one of:",
        ]
        lines += [f"  {spec.vendor} {spec.hardware}: {spec.install}" for spec in missing]
        lines.append("Or run with --execution cpu to accept CPU inference deliberately.")
        super().__init__("\n".join(lines))


class ProviderNotBound(RuntimeError):
    """Raised when the session loaded, but not onto the provider that was requested."""

    def __init__(self, requested: str, actual: Sequence[str]) -> None:
        self.requested = requested
        self.actual = list(actual)
        super().__init__(
            f"Requested {requested}, but the session bound {', '.join(actual) or 'nothing'}. "
            "ONNX Runtime falls back silently when a provider rejects a graph. Check that the "
            "model is quantized and statically shaped for this accelerator."
        )


def installed_providers() -> list[str]:
    """Execution providers this ONNX Runtime build can actually load."""
    return list(_onnxruntime().get_available_providers())


def known_providers() -> list[str]:
    """Every provider name ONNX Runtime recognises, installed or not."""
    return list(_onnxruntime().get_all_providers())


def spec_for(name: str) -> ProviderSpec | None:
    return _BY_NAME.get(name)


def probe() -> list[ProviderStatus]:
    """Registry entries paired with availability in this build."""
    installed = set(installed_providers())
    return [ProviderStatus(spec, spec.name in installed) for spec in PROVIDERS]


def available_npu_providers() -> list[ProviderSpec]:
    """NPU providers this build can load, in preference order."""
    return [status.spec for status in probe() if status.available and status.spec.is_npu]


def resolve(
    policy: str = "npu",
    provider: str | None = None,
    options: Mapping[str, str] | None = None,
) -> list[tuple[str, dict[str, str]]]:
    """Build the ONNX Runtime provider list for a policy.

    Returns ``(name, provider_options)`` pairs in the order the session should
    try them. Raises :class:`NpuUnavailable` under the strict ``npu`` policy
    when nothing suitable is installed, rather than letting the session drop to
    CPU without saying so.
    """
    if policy not in EXECUTION_POLICIES:
        raise ValueError(
            f"Unknown execution policy {policy!r}; expected one of {EXECUTION_POLICIES}"
        )

    if provider:
        return _resolve_pinned(policy, provider, options)

    if policy == "cpu":
        return [(CPU_PROVIDER, {})]

    chosen = [(spec.name, spec.provider_options(options)) for spec in available_npu_providers()]
    if not chosen:
        if policy == "npu":
            raise NpuUnavailable(policy, probe())
        return [(CPU_PROVIDER, {})]

    if policy == "npu-preferred":
        chosen.append((CPU_PROVIDER, {}))
    return chosen


def _resolve_pinned(
    policy: str,
    provider: str,
    options: Mapping[str, str] | None,
) -> list[tuple[str, dict[str, str]]]:
    spec = spec_for(provider)
    if provider not in known_providers():
        raise ValueError(
            f"{provider!r} is not an ONNX Runtime provider. "
            f"Known names: {', '.join(known_providers())}"
        )
    if provider not in installed_providers():
        if spec and spec.is_npu:
            raise NpuUnavailable(policy, probe())
        hint = f" Install it with: {spec.install}" if spec else ""
        raise ValueError(f"{provider!r} is not installed in this build.{hint}")

    resolved = [(provider, spec.provider_options(options) if spec else dict(options or {}))]
    if policy == "npu-preferred" and provider != CPU_PROVIDER:
        resolved.append((CPU_PROVIDER, {}))
    return resolved


def active_provider(session) -> str:
    """The provider a live session actually bound first."""
    providers = list(session.get_providers())
    return providers[0] if providers else "unknown"


def verify_binding(session, requested: Sequence[tuple[str, dict]], policy: str) -> str:
    """Confirm the session bound what the policy asked for, and return the winner.

    ONNX Runtime accepts a provider list and silently drops the entries it
    cannot use, so the only trustworthy check is what the session reports after
    construction.
    """
    active = active_provider(session)
    if policy != "npu":
        return active
    wanted = [name for name, _opts in requested if name != CPU_PROVIDER]
    if active not in wanted:
        raise ProviderNotBound(wanted[0] if wanted else "an NPU provider", session.get_providers())
    return active


def requires_static_shapes(provider: str) -> bool:
    spec = spec_for(provider)
    return bool(spec and spec.static_shapes_required)


def dynamic_axes(shape: Sequence[object]) -> list[int]:
    """Indices of input dimensions ONNX Runtime reports as dynamic."""
    return [i for i, dim in enumerate(shape) if not isinstance(dim, int) or dim <= 0]


def describe_environment() -> dict:
    """Everything ``ppe devices`` needs to explain this host."""
    statuses = probe()
    return {
        "onnxruntime": _onnxruntime().__version__,
        "platform": sys.platform,
        "installed_providers": installed_providers(),
        "npu_available": [s.spec.name for s in statuses if s.available and s.spec.is_npu],
        "providers": [status.as_dict() for status in statuses],
    }


def _qnn_backend_path() -> str:
    return "QnnHtp.dll" if sys.platform == "win32" else "libQnnHtp.so"


def _onnxruntime():
    try:
        import onnxruntime
    except ImportError as exc:
        raise ImportError(
            "NPU inference needs ONNX Runtime. Install the build for your accelerator, "
            "or `pip install onnxruntime` for a CPU-only development install."
        ) from exc
    return onnxruntime
