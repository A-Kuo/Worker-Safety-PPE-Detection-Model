"""Released model registry and verified download (``ppe fetch-model``).

Weights are not in the wheel or in git: they are release assets. The registry packaged with the code
(``models.json``) pins each file's URL, size and SHA-256, so a download is accepted only if it is
byte-for-byte the file that was evaluated in ``docs/MODEL_CARD.md``. Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Callable

CHUNK = 1 << 20
USER_AGENT = "ppe-fetch-model"


class ModelVerificationError(RuntimeError):
    """A downloaded or existing file is not the registered model."""


@dataclass(frozen=True)
class ModelEntry:
    name: str
    file: str
    url: str
    sha256: str
    bytes: int
    conf: float | None = None
    precision: str = ""
    description: str = ""


def load_registry() -> dict[str, Any]:
    text = resources.files("ppe").joinpath("models.json").read_text(encoding="utf-8")
    return json.loads(text)


def entries() -> dict[str, ModelEntry]:
    reg = load_registry()
    return {
        name: ModelEntry(
            name=name,
            file=m["file"],
            url=m["url"],
            sha256=m["sha256"].lower(),
            bytes=int(m["bytes"]),
            conf=m.get("conf"),
            precision=m.get("precision", ""),
            description=m.get("description", ""),
        )
        for name, m in reg["models"].items()
    }


def default_name() -> str:
    return str(load_registry()["default"])


def get_entry(name: str | None = None) -> ModelEntry:
    table = entries()
    key = name or default_name()
    if key not in table:
        raise KeyError(f"Unknown model {key!r}. Known: {', '.join(sorted(table))}")
    return table[key]


def entry_for_sha256(sha256: str | None) -> ModelEntry | None:
    """Registry entry whose file hash is ``sha256`` (how a loaded model is recognised)."""
    if not sha256:
        return None
    return next((e for e in entries().values() if e.sha256 == sha256.lower()), None)


def recommended_conf(sha256: str | None) -> float | None:
    entry = entry_for_sha256(sha256)
    return entry.conf if entry else None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def is_verified(path: Path, entry: ModelEntry) -> bool:
    path = Path(path)
    return path.is_file() and path.stat().st_size == entry.bytes and file_sha256(path) == entry.sha256


def fetch(
    entry: ModelEntry,
    dest_dir: Path,
    *,
    url: str | None = None,
    force: bool = False,
    opener: Callable[..., Any] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[Path, bool]:
    """Download ``entry`` into ``dest_dir`` and verify it. Returns ``(path, downloaded)``.

    - An existing file that already verifies is kept (``downloaded=False``) unless ``force``.
    - The download goes to a temporary file in ``dest_dir`` and is moved into place only after its
      size and SHA-256 match, so an interrupted or tampered download never becomes the model.
    - ``url`` overrides the registry URL (mirror / local file); the hash check still applies.
    """
    dest_dir = Path(dest_dir)
    target = dest_dir / entry.file
    if not force and is_verified(target, entry):
        return target, False
    dest_dir.mkdir(parents=True, exist_ok=True)
    source = url or entry.url
    if not source.lower().startswith(("https://", "http://", "file://")):
        raise ValueError(f"Unsupported URL scheme: {source}")
    open_url = opener or urllib.request.urlopen
    request = urllib.request.Request(source, headers={"User-Agent": USER_AGENT})
    fd, tmp_name = tempfile.mkstemp(prefix=entry.file + ".", suffix=".part", dir=dest_dir)
    tmp = Path(tmp_name)
    try:
        h = hashlib.sha256()
        size = 0
        with os.fdopen(fd, "wb") as out, open_url(request, timeout=60) as resp:
            while block := resp.read(CHUNK):
                size += len(block)
                if size > entry.bytes:
                    raise ModelVerificationError(
                        f"{entry.file}: download is larger than the registered {entry.bytes} bytes; aborted."
                    )
                h.update(block)
                out.write(block)
                if progress:
                    progress(size, entry.bytes)
        if size != entry.bytes:
            raise ModelVerificationError(
                f"{entry.file}: got {size} bytes, expected {entry.bytes} (incomplete or wrong file)."
            )
        digest = h.hexdigest()
        if digest != entry.sha256:
            raise ModelVerificationError(
                f"{entry.file}: SHA-256 mismatch.\n  expected {entry.sha256}\n  got      {digest}\n"
                "The file was not installed."
            )
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target, True
