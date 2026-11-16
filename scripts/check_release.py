#!/usr/bin/env python3
"""Consistency checks before a release is cut. Fails loudly instead of shipping a mismatch.

    python scripts/check_release.py              # internal consistency
    python scripts/check_release.py --tag v0.1.0 # ...and that the tag matches

Checks that the package version, the model registry (``src/ppe/models.json``) and the model card agree:
a tag that does not match the version, a download URL pointing at another release, or a hash in the
registry that is not the one documented in ``docs/MODEL_CARD.md`` would each ship a broken
``ppe fetch-model``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def package_version(root: Path = REPO) -> str:
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    if not match:
        raise SystemExit("pyproject.toml: no version found")
    return match.group(1)


def problems(root: Path = REPO, tag: str | None = None) -> list[str]:
    out: list[str] = []
    version = package_version(root)
    registry = json.loads((root / "src" / "ppe" / "models.json").read_text(encoding="utf-8"))
    card_path = root / "docs" / "MODEL_CARD.md"
    card = card_path.read_text(encoding="utf-8") if card_path.is_file() else ""
    release = registry.get("release", "")

    if release != f"v{version}":
        out.append(f"models.json release {release!r} != package version v{version}")
    if tag is not None and tag != f"v{version}":
        out.append(f"tag {tag!r} != package version v{version}")
    if registry.get("default") not in registry.get("models", {}):
        out.append(f"models.json default {registry.get('default')!r} is not a registered model")

    files = set()
    for name, m in registry.get("models", {}).items():
        sha = str(m.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            out.append(f"{name}: sha256 is not 64 lowercase hex characters")
        if not isinstance(m.get("bytes"), int) or m["bytes"] <= 0:
            out.append(f"{name}: bytes must be a positive integer")
        expected_tail = f"/releases/download/{release}/{m.get('file')}"
        url = str(m.get("url", ""))
        if not url.startswith("https://") or not url.endswith(expected_tail):
            out.append(f"{name}: url must be https and end with {expected_tail}")
        if m.get("file") in files:
            out.append(f"{name}: duplicate file name {m.get('file')}")
        files.add(m.get("file"))
        conf = m.get("conf")
        if conf is not None and not 0.0 < float(conf) < 1.0:
            out.append(f"{name}: conf {conf} outside (0, 1)")
        if card and sha not in card:
            out.append(f"{name}: sha256 {sha[:16]}... is not documented in docs/MODEL_CARD.md")
    if not card:
        out.append("docs/MODEL_CARD.md is missing")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default=None, help="Git tag being released, e.g. v0.1.0")
    args = parser.parse_args(argv)
    found = problems(tag=args.tag)
    for line in found:
        print(f"RELEASE CHECK FAILED: {line}", file=sys.stderr)
    if found:
        return 1
    print(f"release check ok: v{package_version()}" + (f" (tag {args.tag})" if args.tag else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
