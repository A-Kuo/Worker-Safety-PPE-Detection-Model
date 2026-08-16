import random
from pathlib import Path

from PIL import Image

from src.data.dedup import average_hash, find_duplicate_groups, hamming_distance


def _make_textured_image(path: Path, seed: int, jitter: int = 0) -> None:
    """A reproducible pseudo-random-noise image (with optional pixel jitter to
    simulate resave/recompression) - unlike a flat solid color, this actually
    varies around its own mean, which is what average-hash needs to be
    meaningful."""
    rng = random.Random(seed)
    img = Image.new("RGB", (64, 64))
    pixels = [
        (
            min(255, max(0, rng.randint(0, 255) + jitter)),
            min(255, max(0, rng.randint(0, 255) + jitter)),
            min(255, max(0, rng.randint(0, 255) + jitter)),
        )
        for _ in range(64 * 64)
    ]
    img.putdata(pixels)
    img.save(path)


def test_identical_images_have_zero_distance(tmp_path):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    _make_textured_image(a, seed=1)
    _make_textured_image(b, seed=1)
    assert hamming_distance(average_hash(a), average_hash(b)) == 0


def test_very_different_images_have_large_distance(tmp_path):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    _make_textured_image(a, seed=1)
    _make_textured_image(b, seed=2)
    assert hamming_distance(average_hash(a), average_hash(b)) >= 15


def test_find_duplicate_groups_clusters_near_identical_images(tmp_path):
    paths = []
    # img0 and img1 are the "same" underlying image with a small jitter
    # (simulating recompression); img2 is unrelated.
    for i, (seed, jitter) in enumerate([(1, 0), (1, 3), (2, 0)]):
        p = tmp_path / f"img{i}.jpg"
        _make_textured_image(p, seed=seed, jitter=jitter)
        paths.append(p)

    groups = find_duplicate_groups(paths, threshold=10)
    assert len(groups) == 1
    assert set(groups[0].paths) == {paths[0], paths[1]}
