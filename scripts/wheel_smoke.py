#!/usr/bin/env python3
"""Install the built wheel in a fresh virtualenv and use it from OUTSIDE the repository.

This is the check that catches packaging bugs unit tests cannot see (the test suite runs against
``src/``): a path derived from ``__file__`` that points into site-packages, a module or data file
missing from the wheel, a hidden import of torch, a console script that does not start.

    python -m build
    python scripts/wheel_smoke.py                 # newest dist/*.whl
    python scripts/wheel_smoke.py --wheel path/to/ppe-0.1.0-py3-none-any.whl

Needs ``onnx`` in the *calling* interpreter (to generate the test model); the fresh venv gets only
``<wheel>[cpu]``. Works on Linux, macOS and Windows.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path, *, expect: int = 0, both: bool = False) -> str:
    """Run a command; return stdout (JSON-safe), or stdout + stderr with ``both`` (for messages)."""
    proc = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    if proc.returncode != expect:
        sys.stderr.write(f"\n$ {' '.join(str(c) for c in cmd)}\n{proc.stdout}\n{proc.stderr}\n")
        raise SystemExit(f"wheel smoke: expected exit {expect}, got {proc.returncode}")
    return proc.stdout + proc.stderr if both else proc.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wheel", type=Path, default=None)
    args = parser.parse_args()

    wheel = args.wheel
    if wheel is None:
        wheels = sorted((REPO / "dist").glob("ppe-*.whl"), key=lambda p: p.stat().st_mtime)
        if not wheels:
            raise SystemExit("No wheel in dist/. Run `python -m build` first or pass --wheel.")
        wheel = wheels[-1]
    wheel = wheel.resolve()
    print(f"wheel: {wheel.name}")

    sys.path.insert(0, str(REPO / "src"))
    import cv2
    import numpy as np

    from ppe.testing import make_tiny_detector

    with tempfile.TemporaryDirectory(prefix="ppe_wheel_smoke_") as tmp:
        work = Path(tmp).resolve() / "work"
        work.mkdir()
        if REPO in work.parents:
            raise SystemExit(f"temp dir {work} is inside the repo; the smoke must run outside it")
        make_tiny_detector(work / "tiny.onnx")
        cv2.imwrite(str(work / "frame.png"), np.full((96, 128, 3), 114, dtype=np.uint8))

        venv = Path(tmp) / "venv"
        run([sys.executable, "-m", "venv", venv], cwd=work)
        bindir = venv / ("Scripts" if sys.platform == "win32" else "bin")
        py = bindir / ("python.exe" if sys.platform == "win32" else "python")
        ppe = bindir / ("ppe.exe" if sys.platform == "win32" else "ppe")
        run([py, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", f"{wheel}[cpu]"], cwd=work)

        # 1. Installed from the wheel, torch-free, and aware it is not in a source checkout.
        probe = (
            "import importlib.util as u, json, sys, ppe, ppe.cli, ppe.runtime, ppe.window.pipeline, ppe.models\n"
            "from ppe import paths\n"
            "print(json.dumps({'file': ppe.__file__, 'source_root': str(paths.source_root()),\n"
            "  'project_root': str(paths.project_root()), 'torch_loaded': 'torch' in sys.modules,\n"
            "  'torch_installed': u.find_spec('torch') is not None,\n"
            "  'ultralytics_installed': u.find_spec('ultralytics') is not None,\n"
            "  'models': sorted(ppe.models.entries())}))\n"
        )
        info = json.loads(run([py, "-c", probe], cwd=work).strip().splitlines()[-1])
        assert str(REPO) not in info["file"], f"imported from the repo, not the wheel: {info['file']}"
        assert info["source_root"] == "None", info
        assert Path(info["project_root"]) == work, info
        assert not info["torch_loaded"] and not info["torch_installed"] and not info["ultralytics_installed"], info
        assert "vest_specialist" in info["models"], "models.json missing from the wheel"
        print(f"installed: {info['file']}")

        # 2. Console script starts; providers and the registry are readable.
        providers = json.loads(run([ppe, "providers"], cwd=work))
        assert "CPUExecutionProvider" in providers["resolved"], providers["resolved"]
        listed = json.loads(run([ppe, "fetch-model", "--list"], cwd=work))
        assert any(m["default"] for m in listed)

        # 3. Predict with a relative model path and a relative image, cwd outside the repo.
        out = json.loads(run([ppe, "predict", "--weights", "tiny.onnx", "--source", "frame.png", "--imgsz", "64"], cwd=work))
        assert out["counts"] == {"no_vest": 1, "vest": 1}, out["counts"]
        assert "compliance_note" in out
        print(f"predict: {out['counts']}")

        # 4. Default model lookup in the per-user model directory.
        store = work / "store"
        store.mkdir()
        (store / "vest_specialist.onnx").write_bytes((work / "tiny.onnx").read_bytes())
        script = (
            "import os, subprocess, sys\n"
            f"os.environ['PPE_MODEL_DIR'] = r'{store}'\n"
            f"sys.exit(subprocess.call([r'{ppe}', 'predict', '--source', 'frame.png', '--imgsz', '64']))\n"
        )
        run([py, "-c", script], cwd=work)

        # 5. A wrong file must be rejected by fetch-model and must not be installed.
        fake = work / "fake.onnx"
        fake.write_bytes(b"not the released model")
        dest = work / "fetched"
        msg = run([ppe, "fetch-model", "--dest", dest, "--url", fake.as_uri()], cwd=work, expect=1, both=True)
        assert "expected" in msg and not (dest / "vest_specialist.onnx").exists(), msg
        assert not list(dest.glob("*.part")), "partial download left behind"
        print("fetch-model: tampered file rejected")

        # 6. Diagnostics emit JSON; a CPU-only box is never reported as NPU.
        report = json.loads(run([ppe, "verify-npu", "--soft", "--weights", "tiny.onnx", "--imgsz", "64"], cwd=work, expect=1))
        assert report["ok"] is False and report["cpu_resident"], report

        # 7. Missing optional extras give instructions, not tracebacks.
        (work / "w.pt").write_bytes(b"x")
        msg = run([ppe, "export", "--weights", "w.pt", "--out", "o.onnx"], cwd=work, expect=1, both=True)
        assert "ppe[train]" in msg and "Traceback" not in msg, msg
        msg = run([ppe, "window", "--selftest", "--source", "synthetic", "--weights", "tiny.onnx"], cwd=work, expect=1, both=True)
        assert "ppe[gui]" in msg and "Traceback" not in msg, msg

    print("WHEEL_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
