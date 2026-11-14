#!/usr/bin/env python3
"""Fetch and pin an external Hugging Face checkpoint for fine-tuning.

Downloads one file from a Hugging Face Hub repo at an explicit, pinned
revision (commit SHA — never "latest"), copies it into
``models/pretrained/<name>/``, and records provenance (repo_id, filename,
revision, license, sha256, fetched_at, fetched_by) in
``models/pretrained/manifest.json``.

Loading a third-party ``.pt`` file involves ``torch``/pickle deserialization.
Always pass the exact commit SHA you vetted (see ``configs/models/registry.yaml``)
as ``--revision`` — never a mutable ref like "main" — so a later change to the
upstream repo cannot silently swap the weights you already vetted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _common import REPO_ROOT, link_or_copy  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "models" / "pretrained" / "manifest.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="e.g. Hexmon/vyra-yolo-ppe-detection")
    parser.add_argument("--filename", required=True, help="e.g. best.pt")
    parser.add_argument(
        "--revision",
        required=True,
        help="Pinned commit SHA (not a branch/tag). Required — no 'latest' default.",
    )
    parser.add_argument("--name", required=True, help="Local alias, e.g. hexmon_vyra")
    parser.add_argument("--license", default="", help="License string for the manifest, e.g. cc-by-4.0")
    parser.add_argument("--dest", type=Path, default=REPO_ROOT / "models" / "pretrained")
    parser.add_argument("--fetched-by", default="", help="Attribution, e.g. an email")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"checkpoints": {}}


def _write_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8")


def main() -> int:
    args = _parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. pip install 'ppe[hf]' (or huggingface_hub directly)."
        ) from exc

    print(f"Fetching {args.repo_id}/{args.filename} @ {args.revision}")
    downloaded = Path(
        hf_hub_download(repo_id=args.repo_id, filename=args.filename, revision=args.revision)
    )

    dest_dir = args.dest / args.name
    dest_path = dest_dir / args.filename
    if dest_path.exists():
        dest_path.unlink()
    action = link_or_copy(downloaded, dest_path)
    print(f"{action}: {dest_path}")

    sha256 = _sha256(dest_path)
    manifest = _load_manifest()
    manifest.setdefault("checkpoints", {})[args.name] = {
        "repo_id": args.repo_id,
        "filename": args.filename,
        "revision": args.revision,
        "license": args.license,
        "sha256": sha256,
        "local_path": str(dest_path.relative_to(REPO_ROOT)),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fetched_by": args.fetched_by,
    }
    _write_manifest(manifest)
    print(f"sha256={sha256}")
    print(f"Wrote {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
