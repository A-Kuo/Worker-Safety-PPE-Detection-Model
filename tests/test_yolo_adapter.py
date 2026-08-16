from pathlib import Path

from src.evaluation.yolo_adapter import find_image_for_stem, yolo_label_to_ground_truths


def test_yolo_label_to_ground_truths_converts_normalized_to_pixel_coords(tmp_path):
    label_path = tmp_path / "img1.txt"
    # class 2, center (0.5, 0.5), width 0.2, height 0.4, in a 100x200 image
    label_path.write_text("2 0.5 0.5 0.2 0.4\n")

    gts = yolo_label_to_ground_truths(label_path, img_width=100, img_height=200)

    assert len(gts) == 1
    gt = gts[0]
    assert gt.class_id == 2
    x1, y1, x2, y2 = gt.box
    assert x1 == 40.0  # (0.5 - 0.1) * 100
    assert x2 == 60.0  # (0.5 + 0.1) * 100
    assert y1 == 60.0  # (0.5 - 0.2) * 200
    assert y2 == 140.0  # (0.5 + 0.2) * 200


def test_yolo_label_to_ground_truths_handles_multiple_lines(tmp_path):
    label_path = tmp_path / "img2.txt"
    label_path.write_text("0 0.5 0.5 0.2 0.2\n1 0.25 0.25 0.1 0.1\n")
    gts = yolo_label_to_ground_truths(label_path, img_width=100, img_height=100)
    assert len(gts) == 2
    assert {gt.class_id for gt in gts} == {0, 1}


def test_yolo_label_to_ground_truths_missing_file_returns_empty(tmp_path):
    assert yolo_label_to_ground_truths(tmp_path / "does_not_exist.txt", 100, 100) == []


def test_find_image_for_stem_matches_known_extensions(tmp_path):
    (tmp_path / "photo.png").touch()
    found = find_image_for_stem(tmp_path, "photo")
    assert found == tmp_path / "photo.png"


def test_find_image_for_stem_returns_none_when_missing(tmp_path):
    assert find_image_for_stem(tmp_path, "nope") is None
