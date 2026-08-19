"""Schema remapping and dataset YAML helpers."""

from __future__ import annotations

import pytest

from ppe.schema import (
    COMBINED_RAW_TO_UNIFIED,
    CONSTRUCTION_TO_UNIFIED,
    HHU_TO_UNIFIED,
    RAW_TO_UNIFIED,
    SHARED_EVAL_CLASSES,
    UNIFIED_CLASS_NAMES,
    remap_yolo_line,
    unified_id,
    unify_name,
    write_dataset_yaml,
)

CONSTRUCTION_RAW_NAMES = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
    "machinery",
    "vehicle",
]

COMBINED_RAW_NAMES = [
    "Hardhat",
    "NO-Hardhat",
    "Safety Vest",
    "NO-Safety Vest",
    "Goggles",
    "NO-Goggles",
    "Gloves",
    "NO-Gloves",
    "Mask",
    "NO-Mask",
    "Person",
    "Safety Cone",
    "Ladder",
    "Fall-Detected",
]

HHU_RAW_NAMES = ["helmet", "hi-viz helmet", "hi-viz vest", "person", "head"]


def test_unified_class_order():
    assert UNIFIED_CLASS_NAMES == [
        "helmet",
        "no_helmet",
        "vest",
        "no_vest",
        "goggles",
        "no_goggles",
        "gloves",
        "no_gloves",
        "mask",
        "no_mask",
        "person",
        "cone",
        "ladder",
        "fall_detected",
    ]
    assert unified_id("helmet") == 0
    assert unified_id("fall_detected") == 13
    assert len(UNIFIED_CLASS_NAMES) == 14


def test_unified_id_unknown_raises():
    with pytest.raises(KeyError, match="Unknown unified class"):
        unified_id("machinery")


def test_combined_mapping_covers_fourteen_classes():
    assert set(COMBINED_RAW_TO_UNIFIED.values()) == set(UNIFIED_CLASS_NAMES)
    assert COMBINED_RAW_TO_UNIFIED["Hardhat"] == "helmet"
    assert COMBINED_RAW_TO_UNIFIED["Fall-Detected"] == "fall_detected"


def test_construction_omits_machinery_and_vehicle():
    assert "machinery" not in CONSTRUCTION_TO_UNIFIED
    assert "vehicle" not in CONSTRUCTION_TO_UNIFIED
    assert set(CONSTRUCTION_TO_UNIFIED.values()) <= set(UNIFIED_CLASS_NAMES)


def test_hhu_head_is_no_helmet():
    assert HHU_TO_UNIFIED["head"] == "no_helmet"
    assert HHU_TO_UNIFIED["hi-viz helmet"] == "helmet"
    assert HHU_TO_UNIFIED["hi-viz vest"] == "vest"


def test_shared_eval_classes_are_construction_overlap():
    assert SHARED_EVAL_CLASSES == [
        "helmet",
        "no_helmet",
        "vest",
        "no_vest",
        "mask",
        "no_mask",
        "person",
        "cone",
    ]
    assert "goggles" not in SHARED_EVAL_CLASSES
    assert set(SHARED_EVAL_CLASSES) <= set(CONSTRUCTION_TO_UNIFIED.values())


def test_remap_combined_hardhat_to_helmet():
    line = remap_yolo_line("0 0.50 0.40 0.20 0.30", COMBINED_RAW_NAMES, COMBINED_RAW_TO_UNIFIED)
    assert line == "0 0.50 0.40 0.20 0.30"


def test_remap_combined_no_hardhat_preserves_bbox():
    line = remap_yolo_line("1 0.11 0.22 0.33 0.44", COMBINED_RAW_NAMES, COMBINED_RAW_TO_UNIFIED)
    assert line == f"{unified_id('no_helmet')} 0.11 0.22 0.33 0.44"


def test_remap_combined_fall_detected():
    raw_id = COMBINED_RAW_NAMES.index("Fall-Detected")
    line = remap_yolo_line(f"{raw_id} 0.5 0.5 0.1 0.1", COMBINED_RAW_NAMES, COMBINED_RAW_TO_UNIFIED)
    assert line == f"{unified_id('fall_detected')} 0.5 0.5 0.1 0.1"


def test_remap_construction_drops_machinery_and_vehicle():
    machinery = remap_yolo_line(
        "8 0.5 0.5 0.1 0.1", CONSTRUCTION_RAW_NAMES, CONSTRUCTION_TO_UNIFIED
    )
    vehicle = remap_yolo_line("9 0.5 0.5 0.2 0.2", CONSTRUCTION_RAW_NAMES, CONSTRUCTION_TO_UNIFIED)
    assert machinery is None
    assert vehicle is None


def test_remap_construction_safety_vest():
    line = remap_yolo_line("7 0.3 0.4 0.5 0.6", CONSTRUCTION_RAW_NAMES, CONSTRUCTION_TO_UNIFIED)
    assert line == f"{unified_id('vest')} 0.3 0.4 0.5 0.6"


def test_remap_hhu_head_and_hiviz():
    head = remap_yolo_line("4 0.2 0.3 0.1 0.1", HHU_RAW_NAMES, HHU_TO_UNIFIED)
    hiviz = remap_yolo_line("1 0.2 0.3 0.1 0.1", HHU_RAW_NAMES, HHU_TO_UNIFIED)
    vest = remap_yolo_line("2 0.2 0.3 0.1 0.1", HHU_RAW_NAMES, HHU_TO_UNIFIED)
    assert head == f"{unified_id('no_helmet')} 0.2 0.3 0.1 0.1"
    assert hiviz == f"{unified_id('helmet')} 0.2 0.3 0.1 0.1"
    assert vest == f"{unified_id('vest')} 0.2 0.3 0.1 0.1"


def test_remap_empty_and_out_of_range_dropped():
    assert remap_yolo_line("", COMBINED_RAW_NAMES, COMBINED_RAW_TO_UNIFIED) is None
    assert remap_yolo_line("   ", COMBINED_RAW_NAMES, COMBINED_RAW_TO_UNIFIED) is None
    assert (
        remap_yolo_line("99 0.1 0.1 0.1 0.1", COMBINED_RAW_NAMES, COMBINED_RAW_TO_UNIFIED) is None
    )
    assert remap_yolo_line("not-a-line", COMBINED_RAW_NAMES, COMBINED_RAW_TO_UNIFIED) is None


def test_unify_name_passes_unified_names_through():
    assert unify_name("no_helmet") == "no_helmet"
    assert unify_name("person") == "person"


def test_unify_name_maps_each_dataset_vocabulary():
    assert unify_name("Hardhat") == "helmet"
    assert unify_name("NO-Safety Vest") == "no_vest"
    assert unify_name("hi-viz vest") == "vest"
    assert unify_name("head") == "no_helmet"


def test_unify_name_is_case_insensitive():
    assert unify_name("HARDHAT") == "helmet"
    assert unify_name("safety cone") == "cone"


def test_unify_name_leaves_unknown_classes_visible():
    assert unify_name("forklift") == "forklift"


def test_raw_lookup_covers_all_three_datasets():
    for mapping in (COMBINED_RAW_TO_UNIFIED, CONSTRUCTION_TO_UNIFIED, HHU_TO_UNIFIED):
        assert mapping.items() <= RAW_TO_UNIFIED.items()
    assert set(RAW_TO_UNIFIED.values()) <= set(UNIFIED_CLASS_NAMES)


def test_write_dataset_yaml(tmp_path):
    out = tmp_path / "nested" / "data.yaml"
    write_dataset_yaml(
        out,
        train="data/raw/combined/train/images",
        val="data/raw/combined/valid/images",
        test="data/raw/combined/test/images",
        names=UNIFIED_CLASS_NAMES,
    )
    text = out.read_text(encoding="utf-8")
    assert "train: data/raw/combined/train/images" in text
    assert "nc: 14" in text
    assert "  - helmet" in text
    assert "  - fall_detected" in text
    assert text.splitlines().index("names:") < text.splitlines().index("  - helmet")
