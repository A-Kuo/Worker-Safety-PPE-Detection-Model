"""Command line behaviour, driven against the stub backend."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ppe.cli import build_parser, config_from_args, main
from ppe.sources import open_writer

cv2 = pytest.importorskip("cv2", reason="the image and video subcommands need opencv")


@pytest.fixture
def still(tmp_path):
    path = tmp_path / "site.jpg"
    cv2.imwrite(str(path), np.full((120, 160, 3), 90, dtype=np.uint8))
    return path


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "site.mp4"
    writer = open_writer(path, fps=10.0, size=(64, 48))
    for _ in range(20):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    return path


def run(argv: list[str], capsys) -> tuple[int, str]:
    code = main(argv)
    return code, capsys.readouterr().out


def test_classes_lists_the_schema(capsys):
    code, out = run(["classes"], capsys)
    assert code == 0
    assert "helmet" in out
    assert "fall_detected" in out


def test_classes_as_json(capsys):
    _code, out = run(["classes", "--json"], capsys)
    payload = json.loads(out)
    assert payload["count"] == 14
    assert payload["classes"][0] == "helmet"


def test_image_subcommand(still, capsys):
    code, out = run(["image", str(still), "--backend", "stub"], capsys)
    assert code == 0
    assert str(still) in out


def test_image_subcommand_json(still, capsys):
    _code, out = run(["image", str(still), "--backend", "stub", "--json"], capsys)
    payload = json.loads(out)
    assert len(payload) == 1
    assert payload[0]["image"] == str(still)
    assert "compliance" in payload[0]


def test_image_subcommand_writes_annotated_copies(still, tmp_path, capsys):
    out_dir = tmp_path / "annotated"
    _code, _out = run(
        ["image", str(still), "--backend", "stub", "--save", str(out_dir), "--json"], capsys
    )
    assert list(out_dir.glob("*_annotated.jpg"))


def test_image_subcommand_reports_an_empty_directory(tmp_path, capsys):
    code, _out = run(["image", str(tmp_path), "--backend", "stub"], capsys)
    assert code == 1


def test_video_subcommand(clip, capsys):
    code, out = run(["video", str(clip), "--backend", "stub", "--json"], capsys)
    assert code == 0
    report = json.loads(out)
    assert report["frames"] == 20
    assert report["stride"] == 1
    assert "latency" in report


def test_video_subcommand_caps_frames(clip, capsys):
    _code, out = run(
        ["video", str(clip), "--backend", "stub", "--max-frames", "5", "--json"], capsys
    )
    assert json.loads(out)["frames"] == 5


def test_video_subcommand_writes_a_clip(clip, tmp_path, capsys):
    target = tmp_path / "out.mp4"
    _code, out = run(
        ["video", str(clip), "--backend", "stub", "--save", str(target), "--json"], capsys
    )
    assert target.is_file()
    assert json.loads(out)["annotated"] == str(target)


def test_bench_subcommand(capsys):
    code, out = run(
        ["bench", "--backend", "stub", "--frames", "10", "--warmup", "1", "--json"], capsys
    )
    assert code == 0
    report = json.loads(out)
    assert report["frames"] == 10
    assert report["backend"] == "stub"
    assert report["latency"]["frames"] == 10


def test_bench_against_a_still(still, capsys):
    _code, out = run(
        ["bench", "--backend", "stub", "--source", str(still), "--frames", "3", "--json"], capsys
    )
    assert json.loads(out)["frame_shape"] == [120, 160, 3]


def test_watch_stops_after_the_clip_ends(clip, capsys):
    code, out = run(["watch", str(clip), "--backend", "stub", "--report-every", "0"], capsys)
    assert code == 0
    assert "Watching" in out


def test_a_missing_file_is_an_error_not_a_traceback(tmp_path, capsys):
    code = main(["video", str(tmp_path / "nope.mp4"), "--backend", "stub"])
    assert code == 1
    assert "Could not open source" in capsys.readouterr().err


def test_config_from_args_layers_cli_over_env(monkeypatch):
    monkeypatch.setenv("PPE_CONF", "0.5")
    monkeypatch.setenv("PPE_IMGSZ", "320")
    args = build_parser().parse_args(["image", "x.jpg", "--conf", "0.7"])
    config = config_from_args(args)
    assert config.conf == 0.7
    assert config.imgsz == 320


def test_config_from_args_parses_required_ppe():
    args = build_parser().parse_args(["image", "x.jpg", "--required", "helmet,goggles"])
    assert config_from_args(args).required_ppe == ("helmet", "goggles")


def test_unknown_subcommand_exits_with_usage():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["teleport"])
