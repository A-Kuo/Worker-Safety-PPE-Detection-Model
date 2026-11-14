"""Prove where a model actually runs.

Requesting an NPU execution provider is not the same as running on the NPU: ONNX Runtime always
keeps an implicit CPU EP and quietly assigns any op the NPU EP rejects to it. This module answers
the question an on-device team will ask first — *which provider executed each node?* — by running one
profiled inference and reading the per-node ``provider`` ORT records.

Two independent guarantees back the verdict in strict mode:
1. ``session.disable_cpu_ep_fallback=1`` (see ``backends.make_session_options``): ORT refuses to
   build the session if any node would fall back to the CPU EP.
2. The profile below, which lists every executed node with its provider.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ppe.runtime.backends import is_static_shape, open_ort_session, openvino_device_is_npu
from ppe.runtime.config import ExecutionPolicy
from ppe.runtime.providers import list_providers

_CPU = "CPUExecutionProvider"
_KERNEL_SUFFIX = "_kernel_time"


def _npu_ort_names() -> set[str]:
    return {p.ort_name for p in list_providers(tier="npu")}


def parse_profile(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize ORT profiling events into per-provider node counts.

    ORT writes several events per node (kernel time, fences). We count each node once, from its
    ``*_kernel_time`` event when present, otherwise from any node event. ``Memcpy*`` ops are data
    movement ORT inserts at provider boundaries; they are reported separately and never counted as
    CPU-resident compute.
    """
    nodes: dict[str, dict[str, str]] = {}
    kernel_events = []
    other_events = []
    for ev in events:
        if not isinstance(ev, dict) or ev.get("cat") != "Node":
            continue
        args = ev.get("args") or {}
        provider = args.get("provider")
        if not provider:
            continue
        name = str(ev.get("name", ""))
        (kernel_events if name.endswith(_KERNEL_SUFFIX) else other_events).append((name, args))
    chosen = kernel_events or other_events
    for name, args in chosen:
        node = name[: -len(_KERNEL_SUFFIX)] if name.endswith(_KERNEL_SUFFIX) else name
        node = node.removesuffix("_fence_before").removesuffix("_fence_after")
        nodes.setdefault(node, {"op": str(args.get("op_name", "")), "provider": str(args["provider"])})

    compute = {n: v for n, v in nodes.items() if not v["op"].startswith("Memcpy")}
    transfer = {n: v for n, v in nodes.items() if v["op"].startswith("Memcpy")}
    by_provider = Counter(v["provider"] for v in compute.values())
    cpu_resident = [
        {"node": n, "op": v["op"]} for n, v in sorted(compute.items()) if v["provider"] == _CPU
    ]
    npu_names = _npu_ort_names()
    on_npu = sum(c for p, c in by_provider.items() if p in npu_names)
    total = sum(by_provider.values())
    return {
        "nodes_total": total,
        "by_provider": dict(by_provider),
        "npu_nodes": on_npu,
        "npu_fraction": round(on_npu / total, 4) if total else None,
        "cpu_resident": cpu_resident,
        "transfer_nodes": len(transfer),
    }


def apply_device_pin(summary: dict[str, Any], device_pinned: bool | None) -> dict[str, Any]:
    """Make the NPU counts device-aware.

    The OpenVINO EP is NPU-tier, but it also drives the CPU and iGPU. When its device is not pinned to
    the NPU, its nodes are accelerator nodes, not NPU nodes: keep the original count as
    ``accelerator_nodes`` and zero the NPU figures so a report never over-claims.
    """
    out = dict(summary)
    out["accelerator_nodes"] = out["npu_nodes"]
    if device_pinned is False:
        out["npu_nodes"] = 0
        out["npu_fraction"] = 0.0 if out["nodes_total"] else None
    return out


def _zero_input(session: Any, imgsz: int) -> tuple[str, np.ndarray]:
    inp = session.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    if len(shape) == 4 and not is_static_shape(inp.shape):
        shape[2] = shape[3] = imgsz
    return inp.name, np.zeros(shape, dtype=np.float32)


def verify_placement(
    model_path: str | Path, policy: ExecutionPolicy, *, runs: int = 3
) -> dict[str, Any]:
    """Run ``runs`` profiled inferences on zeros and report where every node executed.

    Returns a JSON-serializable report. ``ok`` is True only when every compute node ran on a
    non-CPU provider. In strict mode a session that builds at all already guarantees this (ORT would
    have refused otherwise), so an empty profile is still reported ``ok=True`` with
    ``method="disable_cpu_ep_fallback"``; outside strict mode an empty profile gives ``ok=None``
    (cannot tell) rather than a guess.
    """
    import onnxruntime as ort

    model_path = Path(model_path)
    tmp = tempfile.mkdtemp(prefix="ppe_profile_")
    session, providers, missing = open_ort_session(
        ort, model_path, policy, profile_prefix=os.path.join(tmp, "ort")
    )
    name, tensor = _zero_input(session, policy.imgsz)
    for _ in range(max(1, runs)):
        session.run(None, {name: tensor})
    profile_path = Path(session.end_profiling())
    try:
        events = json.loads(profile_path.read_text(encoding="utf-8"))
    finally:
        try:
            profile_path.unlink()
            os.rmdir(tmp)
        except OSError:
            pass

    summary = parse_profile(events)
    strict = policy.strict_npu
    if summary["nodes_total"]:
        ok: bool | None = not summary["cpu_resident"]
        method = "profile"
    elif strict:
        ok, method = True, "disable_cpu_ep_fallback"
    else:
        ok, method = None, "no_node_profile"
    # "Ran on OpenVINO" is not "ran on the NPU": that EP also drives the CPU and iGPU. The device
    # string is the only thing that pins it, so a non-NPU device can never be reported as proven.
    device = policy.effective_openvino_device
    device_pinned = None
    if "OpenVINOExecutionProvider" in providers:
        device_pinned = openvino_device_is_npu(device)
        if not device_pinned and ok:
            ok = False
    summary = apply_device_pin(summary, device_pinned)
    return {
        "model": str(model_path),
        "strict_npu": strict,
        "providers_requested": providers,
        "providers_active": list(session.get_providers()),
        "providers_missing": missing,
        "openvino_device_type": device,
        "openvino_device_pinned_to_npu": device_pinned,
        "method": method,
        "ok": ok,
        **summary,
    }
