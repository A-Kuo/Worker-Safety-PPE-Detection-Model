"""Package a finished run's deliverables so they cannot be lost on a Kaggle/Colab VM.

Kaggle's Output tab is whatever is left in /kaggle/working when the session ends, and its
"Download All" bulk-zips all of it (raw + processed datasets are several GB). The vest
specialist run on 2026-09-22 passed every gate but its weights were never retrieved. This
script (1) copies the few files that matter into one small directory, (2) writes a MANIFEST
with sizes, hashes and headline metrics, (3) zips it, (4) asserts the essentials exist, and
(5) optionally prunes everything else from the working directory so the bulk zip Kaggle
produces is itself just the deliverables.

    python scripts/package_deliverables.py --exp vest_specialist --out /kaggle/working \
        --log /kaggle/working/train_vest_specialist.log --prune
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

REQUIRED = ("weights/best.pt", "eval_{exp}.json")
MAX_ZIP_MB = 50
PRUNE_ROOTS = (Path("/kaggle/working"), Path("/content"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(exp: str, out_root: Path, log: Path | None = None, runs_root: Path = Path("runs/train"),
            analysis_root: Path = Path("results/analysis")) -> tuple[Path, list[str]]:
    """Copy deliverables into out_root/deliverables_<exp>/. Returns (dir, missing source names)."""
    dest = out_root / f"deliverables_{exp}"
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "weights").mkdir(parents=True)
    run = runs_root / exp
    wanted = [
        (run / "weights" / "best.pt", dest / "weights" / "best.pt"),
        (run / "weights" / "last.pt", dest / "weights" / "last.pt"),
        (run / "results.csv", dest / "results.csv"),
        (run / "args.yaml", dest / "args.yaml"),
        (analysis_root / f"eval_{exp}.json", dest / f"eval_{exp}.json"),
    ]
    if log is not None:
        wanted.append((log, dest / log.name))
    missing = []
    for src, dst in wanted:
        if src.exists():
            shutil.copy(src, dst)
        else:
            missing.append(str(src))
    return dest, missing


def write_manifest(dest: Path, exp: str) -> dict:
    files = {str(p.relative_to(dest)).replace("\\", "/"): {"bytes": p.stat().st_size, "sha256": _sha256(p)}
             for p in sorted(dest.rglob("*")) if p.is_file() and p.name != "MANIFEST.json"}
    manifest = {"exp": exp, "files": files}
    eval_json = dest / f"eval_{exp}.json"
    if eval_json.exists():
        manifest["metrics"] = json.loads(eval_json.read_text(encoding="utf-8")).get("metrics")
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def check_required(dest: Path, exp: str) -> list[str]:
    return [r.format(exp=exp) for r in REQUIRED if not (dest / r.format(exp=exp)).exists()]


def prune(work_dir: Path, keep: set[str]) -> list[str]:
    """Delete everything directly under work_dir except names in keep. Returns removed names."""
    if work_dir.resolve() not in [r.resolve() for r in PRUNE_ROOTS]:
        raise ValueError(f"refusing to prune {work_dir}: only /kaggle/working or /content are allowed")
    removed = []
    for child in work_dir.iterdir():
        if child.name in keep:
            continue
        shutil.rmtree(child) if child.is_dir() else child.unlink()
        removed.append(child.name)
    return removed


def run(exp: str, out_root: Path, log: Path | None = None, do_prune: bool = False, strict: bool = True,
        runs_root: Path = Path("runs/train"), analysis_root: Path = Path("results/analysis")) -> Path:
    dest, missing = collect(exp, out_root, log, runs_root, analysis_root)
    manifest = write_manifest(dest, exp)
    zip_path = Path(shutil.make_archive(str(out_root / dest.name), "zip", root_dir=dest))
    size_mb = zip_path.stat().st_size / 1e6
    print(f"Wrote {zip_path} ({size_mb:.1f} MB)")
    for name, info in manifest["files"].items():
        print(f"  {name}  {info['bytes'] / 1e6:.2f} MB")
    absent = check_required(dest, exp)
    if absent and not strict:
        print(f"NOTE: not yet available: {absent}")
    elif absent:
        raise AssertionError(f"required deliverables missing: {absent} (sources not found: {missing}) - run is NOT done")
    if size_mb > MAX_ZIP_MB:
        raise AssertionError(f"{zip_path} is {size_mb:.0f} MB (> {MAX_ZIP_MB}); something large got copied")
    if do_prune:
        os.chdir(out_root)  # the repo checkout (our cwd) is about to be deleted
        keep = {dest.name, zip_path.name} | ({log.name} if log is not None and log.parent == out_root else set())
        removed = prune(out_root, keep)
        print(f"Pruned from {out_root}: {removed}")
        print(f"Output tab now holds only: {sorted(p.name for p in out_root.iterdir())}")
    print("All expected deliverables present and packaged.")
    return zip_path


KAGGLE_WORKING = Path("/kaggle/working")


def auto_package(exp: str, repo_root: Path, prune_after: bool, strict: bool) -> Path | None:
    """Hook for scripts the notebook already calls (train.py, sanity_check.py).

    The Kaggle notebook cells live in Kaggle's own copy of the .ipynb and cannot be updated from
    GitHub, but scripts/ is re-cloned every run - so packaging is triggered from here. Only acts on
    a real (non-smoke) Kaggle run; PPE_NO_PACKAGE=1 opts out.
    """
    if exp == "smoke" or os.environ.get("PPE_NO_PACKAGE") or not KAGGLE_WORKING.is_dir():
        return None
    log = KAGGLE_WORKING / f"train_{exp}.log"
    return run(exp, KAGGLE_WORKING, log if log.exists() else None, do_prune=prune_after, strict=strict,
               runs_root=repo_root / "runs" / "train", analysis_root=repo_root / "results" / "analysis")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out", type=Path, required=True, help="Directory the Output tab is built from")
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--prune", action="store_true", help="Delete everything else in --out (Kaggle/Colab only)")
    args = ap.parse_args(argv)
    run(args.exp, args.out, args.log, args.prune)
    return 0


if __name__ == "__main__":
    sys.exit(main())
