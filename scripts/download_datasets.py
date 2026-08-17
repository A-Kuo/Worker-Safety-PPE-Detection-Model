#!/usr/bin/env python3
"""Download Combined v4, Hard Hat Universe, and optional Construction v28.

Uses ``ROBOFLOW_API_KEY``. Defaults to **dry-run** (print URLs/commands only).
A full Combined download is ~44k images - only happens with a key **and**
``--execute``.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _common import REPO_ROOT  # noqa: E402

PREFERRED_HHU_VERSION = 26
PREFERRED_HHU_NAME = "no_nulls_plain"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    workspace: str
    project: str
    version: int | None
    dest: Path
    optional: bool
    notes: str

    @property
    def universe_url(self) -> str:
        ver = self.version if self.version is not None else "latest"
        return (
            f"https://universe.roboflow.com/{self.workspace}/{self.project}/dataset/{ver}"
        )

    @property
    def shorthand(self) -> str:
        ver = self.version if self.version is not None else "<version>"
        return f"{self.workspace}/{self.project}/{ver}"

    def cli_command(self) -> str:
        dest = self.dest.as_posix()
        return f"roboflow download -f yolov8 {self.shorthand} -l {dest}"

    def python_snippet(self) -> str:
        ver = self.version if self.version is not None else PREFERRED_HHU_VERSION
        dest = self.dest.as_posix()
        return (
            "from roboflow import Roboflow\n"
            "rf = Roboflow(api_key=os.environ['ROBOFLOW_API_KEY'])\n"
            f"project = rf.workspace({self.workspace!r}).project({self.project!r})\n"
            f"project.version({ver}).download('yolov8', location={dest!r})"
        )


def default_specs() -> list[DatasetSpec]:
    raw = REPO_ROOT / "data" / "raw"
    return [
        DatasetSpec(
            key="combined",
            workspace="roboflow-universe-projects",
            project="personal-protective-equipment-combined-model",
            version=4,
            dest=raw / "combined",
            optional=False,
            notes="Unified train set: 44,002 images, 14 classes, 70/20/10. Do not merge Construction into this folder.",
        ),
        DatasetSpec(
            key="hardhat",
            workspace="universe-datasets",
            project="hard-hat-universe-0dy7t",
            version=PREFERRED_HHU_VERSION,
            dest=raw / "hardhat",
            optional=False,
            notes=(
                f"Held-out helmet-domain eval. Prefer v{PREFERRED_HHU_VERSION} "
                f"({PREFERRED_HHU_NAME}) if the API exposes that name; else latest."
            ),
        ),
        DatasetSpec(
            key="construction",
            workspace="roboflow-universe-projects",
            project="construction-site-safety",
            version=28,
            dest=raw / "construction",
            optional=True,
            notes="Optional local val only. Tiny split (2605/114/82). Never merge into Combined.",
        ),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually download. Requires ROBOFLOW_API_KEY. Refused without both.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print URLs/commands only (default if no key or no --execute).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["combined", "hardhat", "construction"],
        default=None,
    )
    parser.add_argument("--skip-construction", action="store_true")
    parser.add_argument("--hhu-version", type=int, default=None, help="Override Hard Hat Universe version.")
    return parser.parse_args()


def _resolve_hhu_version(project, requested: int | None) -> int:
    if requested is not None:
        return requested
    versions = []
    for attr in ("versions", "list_versions"):
        fn = getattr(project, attr, None)
        if callable(fn):
            try:
                versions = list(fn())
                break
            except Exception:  # noqa: BLE001
                continue
    for ver in versions:
        name = str(getattr(ver, "name", "") or getattr(ver, "version_name", "")).lower()
        number = getattr(ver, "version", None)
        if number is None:
            number = getattr(ver, "id", None)
        if PREFERRED_HHU_NAME in name:
            try:
                return int(number)
            except (TypeError, ValueError):
                return PREFERRED_HHU_VERSION
        if number is not None and int(number) == PREFERRED_HHU_VERSION:
            return PREFERRED_HHU_VERSION
    if versions:
        last = versions[0]
        number = getattr(last, "version", None) or getattr(last, "id", None)
        try:
            return int(number)
        except (TypeError, ValueError):
            pass
    return PREFERRED_HHU_VERSION


def _print_dry_run(specs: list[DatasetSpec], api_key_present: bool) -> None:
    print("DRY RUN - no images will be downloaded.")
    print(f"ROBOFLOW_API_KEY: {'set' if api_key_present else 'not set'}")
    print("To download later: set ROBOFLOW_API_KEY and pass --execute")
    print()
    for spec in specs:
        opt = " (optional)" if spec.optional else ""
        print(f"## {spec.key}{opt}")
        print(f"Universe: {spec.universe_url}")
        print(f"Dest:     {spec.dest}")
        print(f"Notes:    {spec.notes}")
        print("CLI:")
        print(f"  {spec.cli_command()}")
        print("Python:")
        for line in spec.python_snippet().splitlines():
            print(f"  {line}")
        print()


def _download(spec: DatasetSpec, api_key: str, hhu_version: int | None) -> Path:
    from roboflow import Roboflow

    spec.dest.parent.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(spec.workspace).project(spec.project)
    version = spec.version
    if spec.key == "hardhat":
        version = _resolve_hhu_version(project, hhu_version or spec.version)
        print(f"Hard Hat Universe: using version {version} (prefer {PREFERRED_HHU_VERSION} {PREFERRED_HHU_NAME})")
    if version is None:
        raise SystemExit(f"Could not resolve a version for {spec.key}")
    print(f"Downloading {spec.shorthand} -> {spec.dest}")
    dataset = project.version(int(version)).download("yolov8", location=str(spec.dest))
    location = Path(getattr(dataset, "location", spec.dest))
    print(f"Downloaded {spec.key} to {location}")
    return location


def main() -> int:
    args = _parse_args()
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    specs = default_specs()
    if args.hhu_version is not None:
        specs = [
            DatasetSpec(
                key=s.key,
                workspace=s.workspace,
                project=s.project,
                version=args.hhu_version if s.key == "hardhat" else s.version,
                dest=s.dest,
                optional=s.optional,
                notes=s.notes,
            )
            for s in specs
        ]
    if args.only:
        specs = [s for s in specs if s.key in set(args.only)]
    if args.skip_construction:
        specs = [s for s in specs if s.key != "construction"]

    dry_run = args.dry_run or (not args.execute) or (not api_key)
    if args.execute and not api_key:
        print("ROBOFLOW_API_KEY is not set; falling back to dry-run.", file=sys.stderr)
    if args.execute and api_key and args.dry_run:
        print("--dry-run overrides --execute.", file=sys.stderr)
        dry_run = True

    if dry_run:
        _print_dry_run(specs, bool(api_key))
        return 0

    for spec in specs:
        _download(spec, api_key, args.hhu_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
