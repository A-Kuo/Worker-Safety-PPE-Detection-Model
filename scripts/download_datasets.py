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

from _common import REPO_ROOT, load_dotenv  # noqa: E402

load_dotenv()

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


def _http_get_json(url: str, api_key: str, tries: int = 10) -> dict:
    """GET JSON from Roboflow with retries (TLS resets are common on some Windows/Python 3.14 setups)."""
    import time

    import requests

    last: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            response = requests.get(
                url,
                params={"api_key": api_key, "nocache": "true"},
                timeout=90,
            )
            if response.status_code not in (200, 202):
                raise RuntimeError(f"{url} -> HTTP {response.status_code}: {response.text[:400]}")
            return {"_status": response.status_code, **response.json()}
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"  retry {attempt}/{tries}: {type(exc).__name__}")
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"Failed GET {url}") from last


def _resolve_hhu_version(api_key: str, requested: int | None) -> int:
    if requested is not None:
        return requested
    payload = _http_get_json(
        "https://api.roboflow.com/universe-datasets/hard-hat-universe-0dy7t",
        api_key,
    )
    versions = payload.get("versions") or []
    for ver in versions:
        name = str(ver.get("name") or "").lower()
        vid = str(ver.get("id") or "")
        number = vid.rsplit("/", 1)[-1] if vid else None
        if PREFERRED_HHU_NAME in name:
            try:
                return int(number)
            except (TypeError, ValueError):
                return PREFERRED_HHU_VERSION
        if number is not None and str(number) == str(PREFERRED_HHU_VERSION):
            return PREFERRED_HHU_VERSION
    if versions:
        vid = str(versions[0].get("id") or "")
        try:
            return int(vid.rsplit("/", 1)[-1])
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


def _download_zip(link: str, dest_zip: Path, tries: int = 8) -> None:
    import time

    import requests

    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            with requests.get(link, stream=True, timeout=300) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length") or 0)
                written = 0
                tmp = dest_zip.with_suffix(dest_zip.suffix + ".part")
                with tmp.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        written += len(chunk)
                        if total:
                            pct = 100.0 * written / total
                            print(f"\r  zip {pct:5.1f}% [{written}/{total} bytes]", end="", flush=True)
                print()
                if total and written < total:
                    raise RuntimeError(f"incomplete zip: {written}/{total}")
                tmp.replace(dest_zip)
                return
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"\n  zip retry {attempt}/{tries}: {type(exc).__name__}")
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"Failed to download zip {dest_zip.name}") from last


def _download(spec: DatasetSpec, api_key: str, hhu_version: int | None) -> Path:
    import time
    import zipfile

    spec.dest.parent.mkdir(parents=True, exist_ok=True)
    version = spec.version
    if spec.key == "hardhat":
        version = _resolve_hhu_version(api_key, hhu_version or spec.version)
        print(f"Hard Hat Universe: using version {version} (prefer {PREFERRED_HHU_VERSION} {PREFERRED_HHU_NAME})")
    if version is None:
        raise SystemExit(f"Could not resolve a version for {spec.key}")

    yaml_path = spec.dest / "data.yaml"
    if yaml_path.is_file() and any((spec.dest / split / "images").is_dir() for split in ("train", "valid", "test")):
        print(f"Skip {spec.key}: already extracted at {spec.dest}")
        return spec.dest

    export_url = f"https://api.roboflow.com/{spec.workspace}/{spec.project}/{int(version)}/yolov8"
    print(f"Requesting export {spec.workspace}/{spec.project}/{version}/yolov8 -> {spec.dest}")
    link = None
    for _ in range(180):
        payload = _http_get_json(export_url, api_key)
        export = payload.get("export") or {}
        if isinstance(export, dict) and export.get("link"):
            link = export["link"]
            break
        if payload.get("ready") is False or payload.get("_status") == 202:
            progress = payload.get("progress", 0)
            print(f"  export in progress ({progress})")
            time.sleep(2)
            continue
        print(f"  unexpected export payload keys: {list(payload.keys())[:12]}")
        time.sleep(2)
    if not link:
        raise SystemExit(f"No export link for {spec.key} after polling")

    zip_path = spec.dest.parent / f"{spec.key}.zip"
    print(f"Downloading zip for {spec.key}")
    _download_zip(link, zip_path)
    spec.dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name} -> {spec.dest}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(spec.dest)
    print(f"Downloaded {spec.key} to {spec.dest}")
    return spec.dest


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
