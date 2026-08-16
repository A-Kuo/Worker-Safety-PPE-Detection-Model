"""Perceptual-hash near-duplicate detection for merging PPE datasets.

Motivation (see docs/DATASET_NOTES.md): the base "Construction Site Safety"
dataset's own README states its images were partly cloned from the "Personal
Protective Equipment - Combined Model" dataset - i.e. cross-source image overlap
here is confirmed, not hypothetical. Any merge pipeline must dedup before
splitting, or the same photo can end up in one source's train split and
another's test split.

Uses a simple average-hash (aHash) implemented directly on top of Pillow so we
don't need an extra heavyweight dependency for something this small.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

ImageHash = int


def average_hash(image_path: Path | str, hash_size: int = 8) -> ImageHash:
    """Compute a simple average-hash perceptual fingerprint for one image.

    Robust to resizing, mild recompression, and small color shifts - exactly the
    kind of near-duplicate you get from the same source photo re-exported by
    different Roboflow projects at different resolutions/preprocessing.
    """
    with Image.open(image_path) as img:
        gray = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
        pixels = list(gray.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if p > avg else "0" for p in pixels)
    return int(bits, 2)


def hamming_distance(a: ImageHash, b: ImageHash) -> int:
    return bin(a ^ b).count("1")


@dataclass(frozen=True)
class DuplicateGroup:
    paths: tuple[Path, ...]
    max_internal_distance: int


def find_duplicate_groups(
    image_paths: list[Path],
    threshold: int = 5,
    hash_size: int = 8,
) -> list[DuplicateGroup]:
    """Group images whose aHash Hamming distance is <= `threshold`.

    O(n^2) comparisons - fine for the thousands-of-images scale of a single
    merge run; if this ever needs to scale to the full 44k-image Combined Model
    export, bucket by hash prefix first rather than comparing every pair.

    `threshold=5` (out of a max possible 64 for hash_size=8) is a conservative
    starting point that catches resizes/recompression while avoiding false
    positives on genuinely different photos; re-tune by manually reviewing a
    sample of flagged pairs once run against real data.
    """
    hashes: dict[Path, ImageHash] = {}
    for p in image_paths:
        try:
            hashes[p] = average_hash(p, hash_size=hash_size)
        except Exception as exc:  # noqa: BLE001 - corrupt/unreadable image, skip loudly
            print(f"WARNING: could not hash {p}: {exc}")

    paths = list(hashes.keys())
    visited: set[Path] = set()
    groups: list[DuplicateGroup] = []

    for i, p1 in enumerate(paths):
        if p1 in visited:
            continue
        cluster = [p1]
        for p2 in paths[i + 1 :]:
            if p2 in visited:
                continue
            dist = hamming_distance(hashes[p1], hashes[p2])
            if dist <= threshold:
                cluster.append(p2)
        if len(cluster) > 1:
            max_dist = max(
                hamming_distance(hashes[cluster[0]], hashes[q]) for q in cluster[1:]
            )
            groups.append(DuplicateGroup(paths=tuple(cluster), max_internal_distance=max_dist))
            visited.update(cluster)
        else:
            visited.add(p1)

    return groups
