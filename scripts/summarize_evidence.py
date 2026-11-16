#!/usr/bin/env python3
"""Summarize the committed vest-model evidence: accuracy at the operating points and latency ranges.

Reads ``results/analysis/eval_vest_*_clean_*.json`` (from ``ppe eval``) and
``results/bench/lunarlake_vest_*_run*.json`` (from ``ppe bench``), prints the tables quoted in
``docs/MODEL_CARD.md`` and writes ``results/bench/vest_latency_summary.json``. Numbers in the model
card should be regenerated with this script, never typed by hand.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ANALYSIS = REPO / "results" / "analysis"
BENCH = REPO / "results" / "bench"
OPERATING_CONFS = (0.15, 0.30)


def accuracy_rows() -> list[dict]:
    rows = []
    for path in sorted(ANALYSIS.glob("eval_vest_*_clean_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        for cls, pc in d["per_class"].items():
            for conf in OPERATING_CONFS:
                s = next(x for x in pc["sweep"] if x["conf"] == conf)
                rows.append(
                    {
                        "file": path.name,
                        "class": cls,
                        "n": pc["n_gt_boxes"],
                        "ap50": pc["ap50"],
                        "conf": conf,
                        "precision": s["precision"],
                        "precision_ci95": s["precision_ci95"],
                        "recall": s["recall"],
                        "recall_ci95": s["recall_ci95"],
                        "fp_per_100_images": s["fp_per_100_images"],
                        "model_sha256": d["fingerprint"]["model"]["sha256"],
                    }
                )
    return rows


def latency_summary() -> dict:
    runs: dict[str, list[dict]] = collections.defaultdict(list)
    names: dict[str, list[str]] = collections.defaultdict(list)
    for path in sorted(BENCH.glob("lunarlake_vest_*_run*.json")):
        key = path.stem.split("lunarlake_vest_")[1].rsplit("_run", 1)[0]
        runs[key].append(json.loads(path.read_text(encoding="utf-8")))
        names[key].append(path.name)
    if not runs:
        return {}

    def span(values: list[float]) -> list[float]:
        return [min(values), max(values)]

    configs = {}
    for key, bs in runs.items():
        configs[key] = {
            "runs": len(bs),
            "iterations_per_run": bs[0]["iterations"],
            "p50_ms": span([b["latency_ms"]["p50"] for b in bs]),
            "p95_ms": span([b["latency_ms"]["p95"] for b in bs]),
            "p99_ms": span([b["latency_ms"]["p99"] for b in bs]),
            "preprocess_p50_ms": span([b["stages_ms"]["preprocess_ms"]["p50"] for b in bs]),
            "infer_p50_ms": span([b["stages_ms"]["infer_ms"]["p50"] for b in bs]),
            "postprocess_p50_ms": span([b["stages_ms"]["postprocess_ms"]["p50"] for b in bs]),
            "strict_npu": bs[0]["strict_npu"],
            "model_sha256": bs[0]["fingerprint"]["model"]["sha256"],
            "files": names[key],
        }
    fp = next(iter(runs.values()))[0]["fingerprint"]
    ov = fp.get("openvino") or {}
    return {
        "note": "Ranges are min..max of per-run percentiles. Single non-isolated machine: do not "
                "quote p99 as a guarantee.",
        "processor": fp.get("processor"),
        "platform": fp.get("platform"),
        "onnxruntime": f"{fp['onnxruntime']['distribution']} {fp['onnxruntime']['version']}",
        "openvino": ov.get("version"),
        "devices": ov.get("devices"),
        "configs": configs,
    }


def main() -> int:
    for r in accuracy_rows():
        print(
            f"{r['file']:36s} {r['class']:8s} n={r['n']:3d} AP50={r['ap50']:.3f} conf={r['conf']:.2f} "
            f"P={r['precision']:.3f} {r['precision_ci95']} R={r['recall']:.3f} {r['recall_ci95']} "
            f"FP/100img={r['fp_per_100_images']} sha={r['model_sha256'][:12]}"
        )
    summary = latency_summary()
    if summary:
        out = BENCH / "vest_latency_summary.json"
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n{summary['processor']} | {summary['onnxruntime']} | openvino {summary['openvino']}")
        for key, c in summary["configs"].items():
            print(
                f"{key:10s} runs={c['runs']} p50={c['p50_ms']} p95={c['p95_ms']} p99={c['p99_ms']} "
                f"infer={c['infer_p50_ms']} pre={c['preprocess_p50_ms']} post={c['postprocess_p50_ms']}"
            )
        print(f"Wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
