"""The specialist registry, generic data builder and one-command prep must stay consistent."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import make_specialist_data as msd  # noqa: E402
import prepare_specialist as prep  # noqa: E402
import specialists  # noqa: E402
from ppe.schema import UNIFIED_CLASS_NAMES  # noqa: E402


@pytest.mark.parametrize("exp", sorted(specialists.SPECIALISTS))
def test_every_specialist_is_wired_end_to_end(exp):
    spec = specialists.SPECIALISTS[exp]
    # class names are unified names (inference maps model names to unified names by string)
    assert all(c in UNIFIED_CLASS_NAMES for c in spec.classes)
    # train.py knows the experiment and its config points at the dataset the builder writes
    import train

    assert exp in train.EXPERIMENTS
    cfg = (ROOT / "configs" / "train" / f"{exp}.yaml").read_text(encoding="utf-8")
    assert f"data: data/processed/{exp}/data.yaml" in cfg and f"experiment: {exp}" in cfg


def test_helmet_mapping_and_remap():
    mapping = specialists.id_map(specialists.SPECIALISTS["helmet_specialist"], UNIFIED_CLASS_NAMES)
    assert mapping == {UNIFIED_CLASS_NAMES.index("helmet"): 0, UNIFIED_CLASS_NAMES.index("no_helmet"): 1}
    lines = ["0 0.5 0.5 0.1 0.1", "1 0.4 0.4 0.2 0.2", "2 0.3 0.3 0.2 0.3", "10 0.5 0.5 0.9 0.9"]
    assert msd.remap_label_lines(lines, mapping) == ["0 0.5 0.5 0.1 0.1", "1 0.4 0.4 0.2 0.2"]


def test_vest_wrapper_still_exposes_the_original_api():
    import make_vest_specialist_data as legacy

    assert legacy.remap_label_lines(["2 0.4 0.4 0.2 0.3", "0 0.1 0.1 0.1 0.1"]) == ["0 0.4 0.4 0.2 0.3"]
    mapping = specialists.id_map(specialists.SPECIALISTS["vest_specialist"], UNIFIED_CLASS_NAMES)
    assert mapping == legacy.UNIFIED_TO_SPECIALIST


def test_prepare_steps_only_use_the_specialists_own_data():
    helmet = prep.build_steps("helmet_specialist")
    flat = [" ".join(s) for s in helmet]
    assert any("--classes helmet no_helmet" in s and "data/raw/helmet_annotated" in s for s in flat)
    assert not any("gap_vest" in s for s in flat), "helmet has no external dataset"
    vest = [" ".join(s) for s in prep.build_steps("vest_specialist")]
    assert any("--only combined gap_vest" in s for s in vest)
    assert any("--mapping gap_vest" in s for s in vest)
    # every builder step names the experiment, and downloads can be skipped for local reuse
    assert helmet[-1][-2:] == ["--exp", "helmet_specialist"]
    assert not any("download_datasets" in " ".join(s) for s in prep.build_steps("helmet_specialist", skip_download=True))


def test_check_counts_fails_loudly_on_a_bad_build(tmp_path):
    for split, n in (("train", 600), ("valid", 120), ("test", 5)):
        d = tmp_path / "data" / "processed" / "helmet_specialist" / split / "images"
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"{i}.jpg").write_bytes(b"x")
    with pytest.raises(AssertionError, match="looks wrong"):
        prep.check_counts("helmet_specialist", tmp_path)
