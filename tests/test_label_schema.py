import pytest

from src.data import label_schema


def test_unified_class_ids_are_contiguous_and_unique():
    ids = [label_schema.UNIFIED_CLASS_TO_ID[name] for name in label_schema.UNIFIED_CLASSES]
    assert ids == list(range(len(label_schema.UNIFIED_CLASSES)))


@pytest.mark.parametrize("source_key", list(label_schema.SOURCES))
def test_every_mapped_class_targets_a_real_unified_class(source_key):
    source = label_schema.SOURCES[source_key]
    for source_class, unified_class in source.class_map.items():
        assert unified_class in label_schema.UNIFIED_CLASS_TO_ID, (
            f"{source_key}.{source_class!r} maps to unknown unified class {unified_class!r}"
        )


def test_construction_site_safety_matches_known_baseline_class_list():
    # Locks in the 10-class list confirmed in docs/BASELINE_METRICS.md /
    # docs/assets/baseline/data.yaml - if this ever fails, the upstream dataset
    # version changed and the mapping needs re-review, not a silent update.
    expected = {
        "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
        "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle",
    }
    assert set(label_schema.SOURCES["construction_site_safety"].class_map) == expected


def test_remap_source_labels_happy_path():
    class_names = ["Hardhat", "Person", "machinery"]
    ids = label_schema.remap_source_labels("construction_site_safety", class_names)
    assert ids == [
        label_schema.UNIFIED_CLASS_TO_ID["helmet"],
        label_schema.UNIFIED_CLASS_TO_ID["person"],
        label_schema.UNIFIED_CLASS_TO_ID["machinery"],
    ]


def test_remap_source_labels_raises_on_unmapped_class():
    with pytest.raises(KeyError):
        label_schema.remap_source_labels("construction_site_safety", ["SomeBrandNewClass"])


def test_construction_ppe_none_class_maps_to_no_vest():
    # See docs/DATASET_NOTES.md: verified by drawing this dataset's own boxes -
    # "none" boxes sit over the torso region alongside "no_helmet" over the head,
    # i.e. it's this dataset's (oddly-named) no-vest negative.
    assert label_schema.SOURCES["ultralytics_construction_ppe"].class_map["none"] == "no_vest"
