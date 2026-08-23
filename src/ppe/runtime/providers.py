"""Vendor-agnostic ONNX Runtime execution-provider registry.

NPU / accelerator targeting is expressed as named providers. Availability is
probed at runtime from ``onnxruntime.get_available_providers()``. Torch is not
an EP — it is a separate backend gated by ``allow_torch``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ExecutionProvider:
    """One ORT execution provider entry."""

    key: str
    ort_name: str
    vendor: str
    tier: str  # npu | gpu | cpu
    notes: str = ""

    def options(self) -> dict:
        """Default provider options. Override per vendor when needed."""
        return {}


# Registry order is preference for auto-select when no pin is set:
# NPUs first, then discrete GPUs, then CPU.
PROVIDER_REGISTRY: tuple[ExecutionProvider, ...] = (
    ExecutionProvider(
        key="qnn",
        ort_name="QNNExecutionProvider",
        vendor="Qualcomm",
        tier="npu",
        notes="Snapdragon / Hexagon via ORT QNN EP.",
    ),
    ExecutionProvider(
        key="openvino",
        ort_name="OpenVINOExecutionProvider",
        vendor="Intel",
        tier="npu",
        notes="Intel NPU / iGPU / CPU via OpenVINO EP.",
    ),
    ExecutionProvider(
        key="coreml",
        ort_name="CoreMLExecutionProvider",
        vendor="Apple",
        tier="npu",
        notes="Apple Neural Engine when available.",
    ),
    ExecutionProvider(
        key="cann",
        ort_name="CANNExecutionProvider",
        vendor="Huawei",
        tier="npu",
        notes="Ascend NPU via Huawei CANN.",
    ),
    ExecutionProvider(
        key="tensorrt",
        ort_name="TensorrtExecutionProvider",
        vendor="NVIDIA",
        tier="gpu",
        notes="TensorRT EP when installed with ORT-GPU+TRT.",
    ),
    ExecutionProvider(
        key="cuda",
        ort_name="CUDAExecutionProvider",
        vendor="NVIDIA",
        tier="gpu",
        notes="CUDA EP (discrete GPU, not NPU).",
    ),
    ExecutionProvider(
        key="dml",
        ort_name="DmlExecutionProvider",
        vendor="Microsoft",
        tier="gpu",
        notes="DirectML on Windows (GPU / some NPUs).",
    ),
    ExecutionProvider(
        key="cpu",
        ort_name="CPUExecutionProvider",
        vendor="generic",
        tier="cpu",
        notes="Always-available fallback.",
    ),
)

_BY_KEY = {p.key: p for p in PROVIDER_REGISTRY}
_BY_ORT = {p.ort_name: p for p in PROVIDER_REGISTRY}


def get_provider(key: str) -> ExecutionProvider:
    try:
        return _BY_KEY[key.lower()]
    except KeyError as exc:
        known = ", ".join(sorted(_BY_KEY))
        raise KeyError(f"Unknown provider {key!r}. Known: {known}") from exc


def list_providers(*, tier: str | None = None) -> list[ExecutionProvider]:
    if tier is None:
        return list(PROVIDER_REGISTRY)
    return [p for p in PROVIDER_REGISTRY if p.tier == tier]


def available_ort_names() -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError:
        return []
    return list(ort.get_available_providers())


def resolve_providers(
    preferred: Sequence[str] | None = None,
    *,
    npu_only: bool = False,
    allow_cpu_fallback: bool = True,
) -> list[str]:
    """Return ORT provider name list for ``InferenceSession``.

    ``preferred`` are registry keys (``qnn``, ``openvino``, …) or raw ORT names.
    When empty, auto-selects the first available NPU, else GPU, else CPU.
    """
    installed = available_ort_names()
    installed_set = set(installed)

    def _normalize(token: str) -> ExecutionProvider | None:
        t = token.strip()
        if not t:
            return None
        if t in _BY_KEY:
            return _BY_KEY[t]
        if t in _BY_ORT:
            return _BY_ORT[t]
        # Accept raw ORT names not in our registry.
        return ExecutionProvider(key=t.lower(), ort_name=t, vendor="unknown", tier="gpu")

    chosen: list[str] = []
    if preferred:
        for token in preferred:
            ep = _normalize(token)
            if ep is None:
                continue
            if ep.ort_name not in installed_set:
                continue
            if npu_only and ep.tier not in ("npu", "cpu"):
                continue
            if ep.ort_name not in chosen:
                chosen.append(ep.ort_name)
    else:
        # Auto: first available NPU, else first GPU. CPU is soft-fallback below.
        for tier in (("npu",) if npu_only else ("npu", "gpu")):
            hit = next(
                (ep.ort_name for ep in list_providers(tier=tier) if ep.ort_name in installed_set),
                None,
            )
            if hit:
                chosen.append(hit)
                break

    if allow_cpu_fallback and "CPUExecutionProvider" in installed_set:
        if "CPUExecutionProvider" not in chosen:
            chosen.append("CPUExecutionProvider")
    if not chosen and "CPUExecutionProvider" in installed_set:
        chosen = ["CPUExecutionProvider"]
    return chosen


def provider_status() -> list[dict]:
    """Human-readable status table for docs / CLI."""
    installed = set(available_ort_names())
    rows = []
    for ep in PROVIDER_REGISTRY:
        rows.append(
            {
                "key": ep.key,
                "ort_name": ep.ort_name,
                "vendor": ep.vendor,
                "tier": ep.tier,
                "available": ep.ort_name in installed,
                "notes": ep.notes,
            }
        )
    return rows


def filter_npu_keys(keys: Iterable[str]) -> list[str]:
    out = []
    for key in keys:
        ep = _BY_KEY.get(key.lower()) or _BY_ORT.get(key)
        if ep and ep.tier == "npu":
            out.append(ep.key)
    return out
