"""Latency statistics shared by ``ppe bench`` and the app/window HUD. numpy only, no torch."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

STAGES = ("preprocess_ms", "infer_ms", "postprocess_ms")


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    """Count, mean, min/p50/p90/p95/p99/max of a latency sample (milliseconds).

    Tail percentiles on a handful of samples are noise, so ``n`` is always returned alongside; callers
    should not quote p99 from fewer than ~100 samples.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    p50, p90, p95, p99 = np.percentile(arr, [50, 90, 95, 99])
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 3),
        "min": round(float(arr.min()), 3),
        "p50": round(float(p50), 3),
        "p90": round(float(p90), 3),
        "p95": round(float(p95), 3),
        "p99": round(float(p99), 3),
        "max": round(float(arr.max()), 3),
        "std": round(float(arr.std()), 3),
    }


def fps_from_ms(ms: float) -> float:
    return round(1000.0 / ms, 2) if ms > 0 else 0.0


def summarize_stages(samples: Sequence[Mapping[str, float]]) -> dict[str, dict]:
    """Per-stage summaries from a list of ``{stage: ms}`` dicts (missing stages are skipped)."""
    out: dict[str, dict] = {}
    keys: list[str] = []
    for s in samples:
        for k in s:
            if k not in keys:
                keys.append(k)
    ordered = [k for k in STAGES if k in keys] + [k for k in keys if k not in STAGES]
    for k in ordered:
        out[k] = summarize(s[k] for s in samples if k in s)
    return out
