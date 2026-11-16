"""Runtime evaluator, evidence hygiene and the INT8 recipe (PR 2). No dataset or hardware needed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ppe.compliance import Detection
from ppe.evaluate import (
    iou_one_to_many,
    label_path_for,
    read_yolo_labels,
    score_dataset,
    summarize_scores,
    wilson_interval,
)

REPO = Path(__file__).resolve().parents[1]
# 101-point interpolation (the Ultralytics convention) scores a perfect detector 0.995, not 1.0.
AP_CEILING = 0.995


def D(cls, conf, box):
    return Detection(cls_name=cls, conf=conf, xyxy=tuple(float(v) for v in box))


# --- statistics ---------------------------------------------------------------------------------

def test_wilson_interval_known_values():
    lo, hi = wilson_interval(35, 43)  # the no_vest recall case
    assert lo == pytest.approx(0.6738, abs=1e-3) and hi == pytest.approx(0.9026, abs=1e-3)
    assert wilson_interval(0, 0) is None
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0 and 0.6 < lo < 0.8  # 10/10 does not prove 100%
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0.2 < hi < 0.4


def test_wilson_interval_narrows_with_more_data():
    small = wilson_interval(8, 10)
    large = wilson_interval(800, 1000)
    assert (small[1] - small[0]) > 3 * (large[1] - large[0])


def test_iou():
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=float)
    out = iou_one_to_many((0, 0, 10, 10), boxes)
    assert out[0] == pytest.approx(1.0) and out[1] == pytest.approx(1 / 3) and out[2] == 0.0
    assert len(iou_one_to_many((0, 0, 1, 1), np.zeros((0, 4)))) == 0


# --- labels -------------------------------------------------------------------------------------

def test_read_yolo_labels_and_label_path(tmp_path):
    img = tmp_path / "ds" / "test" / "images" / "a.jpg"
    lbl = label_path_for(img)
    assert lbl == tmp_path / "ds" / "test" / "labels" / "a.txt"
    lbl.parent.mkdir(parents=True)
    lbl.write_text("0 0.5 0.5 0.2 0.4\n1 0.25 0.25 0.5 0.5\n9 0.5 0.5 0.1 0.1\nbad line\n")
    out = read_yolo_labels(lbl, 100, 200, ["vest", "no_vest"])
    assert [c for c, _ in out] == ["vest", "no_vest"]  # out-of-range class id and junk are skipped
    assert out[0][1] == pytest.approx((40, 60, 60, 140))
    assert read_yolo_labels(tmp_path / "missing.txt", 10, 10, ["vest"]) == []  # image with no objects


def test_label_path_uses_the_last_images_component():
    p = Path("root") / "images" / "set" / "images" / "x.png"
    assert label_path_for(p) == Path("root") / "images" / "set" / "labels" / "x.txt"


# --- matching and metrics -----------------------------------------------------------------------

def test_perfect_predictions_score_one():
    gt = [[("vest", (0, 0, 10, 10)), ("no_vest", (20, 20, 30, 30))]]
    preds = [[D("vest", 0.9, (0, 0, 10, 10)), D("no_vest", 0.8, (20, 20, 30, 30))]]
    s = summarize_scores(score_dataset(preds, gt, ["vest", "no_vest"]), 1)
    assert s["map50"] == pytest.approx(AP_CEILING, abs=1e-6)
    row = s["per_class"]["vest"]["sweep"][0]
    assert (row["tp"], row["fp"], row["fn"]) == (1, 0, 0)


def test_duplicate_detection_of_one_object_is_a_false_positive():
    gt = [[("vest", (0, 0, 10, 10))]]
    preds = [[D("vest", 0.9, (0, 0, 10, 10)), D("vest", 0.8, (1, 0, 10, 10))]]
    score = score_dataset(preds, gt, ["vest"])["vest"]
    assert list(score.tp) == [1, 0]
    assert score.at(0.5, 1)["fp"] == 1


def test_wrong_class_and_low_iou_do_not_match():
    gt = [[("vest", (0, 0, 10, 10))]]
    preds = [[D("no_vest", 0.9, (0, 0, 10, 10)), D("vest", 0.8, (8, 8, 30, 30))]]
    scores = score_dataset(preds, gt, ["vest", "no_vest"])
    assert scores["vest"].at(0.1, 1)["tp"] == 0 and scores["vest"].at(0.1, 1)["fn"] == 1
    assert scores["no_vest"].at(0.1, 1)["fp"] == 1 and scores["no_vest"].n_gt == 0
    assert scores["no_vest"].ap50() is None  # no ground truth: AP is undefined, not 0


def test_threshold_sweep_counts_and_rates():
    gt = [[("vest", (0, 0, 10, 10)), ("vest", (20, 0, 30, 10))], []]
    preds = [
        [D("vest", 0.9, (0, 0, 10, 10)), D("vest", 0.3, (20, 0, 30, 10))],
        [D("vest", 0.6, (0, 0, 5, 5))],  # false positive on an image with no vest
    ]
    score = score_dataset(preds, gt, ["vest"])["vest"]
    hi = score.at(0.5, 2)
    assert (hi["tp"], hi["fp"], hi["fn"]) == (1, 1, 1)
    assert hi["precision"] == 0.5 and hi["recall"] == 0.5 and hi["fp_per_100_images"] == 50.0
    lo = score.at(0.2, 2)
    assert (lo["tp"], lo["fp"], lo["fn"]) == (2, 1, 0) and lo["recall"] == 1.0
    assert lo["recall_ci95"][0] < 1.0  # 2/2 still carries an interval
    none = score.at(0.95, 2)
    assert none["precision"] is None and none["recall"] == 0.0  # no detections: precision undefined
    assert 0.5 < score.ap50() < 1.0


def test_ap_is_zero_when_nothing_is_detected():
    score = score_dataset([[]], [[("vest", (0, 0, 10, 10))]], ["vest"])["vest"]
    assert score.ap50() == 0.0 and score.at(0.25, 1)["fn"] == 1


def test_score_dataset_requires_matching_lengths():
    with pytest.raises(ValueError):
        score_dataset([[]], [], ["vest"])


# --- `ppe eval` end to end with a fake session ---------------------------------------------------

def test_cli_eval_writes_counts_protocol_and_scrubbed_paths(tmp_path, monkeypatch, capsys):
    cv2 = pytest.importorskip("cv2")
    from ppe import cli

    monkeypatch.setenv("PPE_HOME", str(tmp_path))
    monkeypatch.setenv("PPE_CLASS_CONF", "no_vest=0.9")  # must NOT influence an evaluation
    images = tmp_path / "ds" / "images"
    labels = tmp_path / "ds" / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    for i in range(3):
        cv2.imwrite(str(images / f"{i}.png"), np.zeros((100, 100, 3), dtype=np.uint8))
        (labels / f"{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (images / "broken.png").write_bytes(b"nope")

    class _Backend:
        name = "fake"
        class_names = {0: "vest", 1: "no_vest"}

        def info(self):
            return {"providers_active": ["CPUExecutionProvider"], "fingerprint": {"model": {"sha256": "abc"}}}

    class _Session:
        backend = _Backend()
        specialist_backends: list = []
        model_path = tmp_path / "m.onnx"

        def __init__(self, policy):
            self.policy = policy

        def predict(self, image):
            return [D("vest", 0.8, (40, 40, 60, 60)), D("no_vest", 0.4, (0, 0, 10, 10))]

    seen = {}

    def fake_open(weights, policy=None, repo_root=None):
        seen["policy"] = policy
        return _Session(policy)

    monkeypatch.setattr("ppe.runtime.open_session", fake_open)
    out = tmp_path / "eval.json"
    assert cli.main(["eval", "--images", "ds/images", "--out", str(out), "--tag", "t"]) == 0
    assert seen["policy"].conf == 0.001 and seen["policy"].class_conf == {}
    r = json.loads(out.read_text())
    assert r["n_images"] == 3 and r["unreadable_images"] == 1
    assert r["label_names"] == ["vest", "no_vest"]
    assert r["per_class"]["vest"]["n_gt_boxes"] == 3 and r["per_class"]["vest"]["ap50"] == pytest.approx(AP_CEILING, abs=1e-4)
    assert r["per_class"]["no_vest"]["n_gt_boxes"] == 0 and r["per_class"]["no_vest"]["ap50"] is None
    assert r["protocol"]["conf_floor"] == 0.001 and r["protocol"]["match_iou"] == 0.5
    assert r["images_dir"] == "<repo>/ds/images" and r["model"] == "<repo>/m.onnx"
    assert str(tmp_path) not in out.read_text().replace("\\\\", "\\")


def test_predict_reports_frame_counts_and_explains_empty_compliance(tmp_path, monkeypatch, capsys):
    cv2 = pytest.importorskip("cv2")
    from ppe import cli

    img = tmp_path / "f.png"
    cv2.imwrite(str(img), np.zeros((50, 50, 3), dtype=np.uint8))

    class _Backend:
        class_names = {0: "vest", 1: "no_vest"}

    class _Session:
        backend = _Backend()
        specialist_backends: list = []
        model_path = tmp_path / "m.onnx"

        def predict_and_comply(self, image):
            return [D("vest", 0.9, (0, 0, 5, 5)), D("vest", 0.8, (10, 10, 20, 20)), D("no_vest", 0.5, (30, 30, 40, 40))], []

    monkeypatch.setattr("ppe.runtime.open_session", lambda *a, **k: _Session())
    assert cli.main(["predict", "--source", str(img)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["counts"] == {"no_vest": 1, "vest": 2}
    assert out["compliance"] == [] and "no `person` class" in out["compliance_note"]

    _Backend.class_names = {0: "Person", 1: "Safety Vest"}  # raw names map to unified `person`
    assert cli.main(["predict", "--source", str(img)]) == 0
    assert "compliance_note" not in json.loads(capsys.readouterr().out)


# --- evidence hygiene ---------------------------------------------------------------------------

def test_scrub_paths_handles_repo_home_and_leaves_lookalikes(tmp_path):
    from ppe.runtime.fingerprint import scrub_paths

    home = Path.home()
    data = {
        "in_repo": str(tmp_path / "models" / "m.onnx"),
        "repo_itself": str(tmp_path),
        "in_home": str(home / ".ppe" / "models" / "m.onnx"),
        "lookalike": str(tmp_path) + "-other/x",  # shares the prefix text but is a different dir
        "plain": "CPUExecutionProvider",
        "n": 3,
    }
    out = scrub_paths(data, tmp_path)
    assert out["in_repo"] == "<repo>/models/m.onnx" and out["repo_itself"] == "<repo>"
    assert out["in_home"] == "~/.ppe/models/m.onnx"
    assert out["lookalike"] == data["lookalike"] and out["plain"] == "CPUExecutionProvider" and out["n"] == 3


def test_committed_vest_evidence_is_consistent_and_path_free():
    analysis = REPO / "results" / "analysis"
    files = sorted(analysis.glob("eval_vest_*_clean_*.json")) + sorted(analysis.glob("verify_npu_vest_*.json"))
    if not files:
        pytest.skip("evidence not generated in this checkout")
    hashes: dict[str, set] = {}
    for f in files:
        text = f.read_text(encoding="utf-8")
        assert ":\\\\" not in text and "/mnt/" not in text and "/home/" not in text, f.name
        d = json.loads(text)
        sha = d["fingerprint"]["model"]["sha256"]
        assert sha and len(sha) == 64
        kind = "int8" if "int8" in f.name else "fp32"
        hashes.setdefault(kind, set()).add(sha)
        if f.name.startswith("eval_"):
            assert d["n_images"] == 145 and d["unreadable_images"] == 0
            assert d["per_class"]["vest"]["n_gt_boxes"] == 249 and d["per_class"]["no_vest"]["n_gt_boxes"] == 43
        else:
            assert d["ok"] is True and d["strict_npu"] is True and d["cpu_resident"] == []
    assert all(len(v) == 1 for v in hashes.values()), hashes  # one model file per variant


# --- validate_accuracy ---------------------------------------------------------------------------

def _va():
    scripts = str(REPO / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import validate_accuracy

    return validate_accuracy


def _runtime_eval(tmp_path, p, r, n, p_ci, r_ci):
    path = tmp_path / "e.json"
    sweep = [{"conf": 0.30, "precision": p, "recall": r, "precision_ci95": p_ci, "recall_ci95": r_ci},
             {"conf": 0.70, "precision": 0.99, "recall": 0.10, "precision_ci95": [0.9, 1], "recall_ci95": [0.05, 0.2]}]
    path.write_text(json.dumps({"per_class": {"no_vest": {"n_gt_boxes": n, "ap50": 0.7, "sweep": sweep}}}))
    return path


def test_validate_reads_runtime_format_at_the_operating_conf(tmp_path):
    va = _va()
    path = _runtime_eval(tmp_path, 0.65, 0.81, 43, [0.51, 0.76], [0.67, 0.90])
    row = va.load_per_class(path)["no_vest"]
    assert row["n"] == 43 and row["conf"] == 0.30 and row["precision"] == 0.65
    assert va.load_per_class(path, conf=0.7)["no_vest"]["recall"] == 0.10
    _, _, failures = va.evaluate(va.load_per_class(path))
    assert "n=43" in failures[0] and "bar excluded by the interval" in failures[0]


def test_validate_says_when_the_sample_cannot_settle_the_bar(tmp_path):
    va = _va()
    path = _runtime_eval(tmp_path, 0.93, 0.93, 43, [0.81, 0.98], [0.81, 0.98])
    _, _, failures = va.evaluate(va.load_per_class(path))
    assert "too few boxes to confirm or exclude the bar" in failures[0]
    path = _runtime_eval(tmp_path, 0.99, 0.99, 5000, [0.985, 0.993], [0.985, 0.993])
    passes, _, _ = va.evaluate(va.load_per_class(path))
    assert "bar supported by the interval" in passes[0]


def test_validate_report_mode_never_fails_but_default_does(tmp_path, capsys):
    va = _va()
    path = _runtime_eval(tmp_path, 0.65, 0.81, 43, [0.51, 0.76], [0.67, 0.90])
    assert va.main([str(path)]) == 1
    assert va.main([str(path), "--report"]) == 0
    assert "report mode" in capsys.readouterr().out


def test_validate_defaults_exclude_the_contaminated_eval():
    va = _va()
    assert not any("unified_order" in p.name for p in va.default_eval_paths())


# --- INT8 recipe ---------------------------------------------------------------------------------

def _yolo_like_model(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    w0 = numpy_helper.from_array(rng.normal(size=(4, 3, 1, 1)).astype("float32"), "w0")
    w1 = numpy_helper.from_array(rng.normal(size=(4, 4, 1, 1)).astype("float32"), "w1")
    nodes = [
        helper.make_node("Conv", ["images", "w0"], ["a"], name="/model.0/conv/Conv"),
        helper.make_node("Conv", ["a", "w1"], ["b"], name="/model.22/cv2/Conv"),
        helper.make_node("Conv", ["b", "w1"], ["c"], name="/model.22/dfl/conv/Conv"),
        helper.make_node("Sigmoid", ["c"], ["d"], name="/model.22/Sigmoid"),
        helper.make_node("Concat", ["d", "b"], ["out"], name="/model.22/Concat_5", axis=1),
    ]
    x = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])
    y = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 8, 32, 32])
    model = helper.make_model(helper.make_graph(nodes, "g", [x], [y], initializer=[w0, w1]),
                              opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "yolo_like.onnx"
    onnx.save(model, str(path))
    return path


def test_head_float_nodes_keep_head_convs_but_not_dfl_or_decode(tmp_path):
    from ppe.tools.quantize import yolo_head_float_nodes

    nodes = yolo_head_float_nodes(_yolo_like_model(tmp_path))
    assert set(nodes) == {"/model.22/dfl/conv/Conv", "/model.22/Sigmoid", "/model.22/Concat_5"}


def test_float_head_quantization_leaves_the_head_unquantized(tmp_path):
    pytest.importorskip("onnxruntime")
    cv2 = pytest.importorskip("cv2")
    import onnx

    from ppe.tools.quantize import quantize_model

    model = _yolo_like_model(tmp_path)
    calib = tmp_path / "calib"
    calib.mkdir()
    rng = np.random.default_rng(1)
    for i in range(4):
        cv2.imwrite(str(calib / f"{i}.png"), rng.integers(0, 255, (24, 40, 3), dtype=np.uint8))
    out = quantize_model(model, tmp_path / "q.onnx", calibration=calib, static=True, imgsz=32,
                         qdq=True, per_channel=True, seed=0, float_head=True)
    graph = onnx.load(str(out)).graph
    ops = [n.op_type for n in graph.node]
    assert "QuantizeLinear" in ops and "DequantizeLinear" in ops  # the body is quantized
    by_output = {o: n for n in graph.node for o in n.output}
    final = by_output["out"]
    assert final.op_type == "Concat"  # the model output is float, not re-quantized
    assert by_output[final.input[0]].op_type == "Sigmoid"  # scores reach the concat without a Q/DQ pair
    sigmoid = by_output[final.input[0]]
    assert by_output[sigmoid.input[0]].op_type == "Conv"  # the excluded DFL conv stayed a float Conv


def test_float_head_requires_static_and_a_recognisable_head(tmp_path):
    from ppe.tools.quantize import quantize_model

    model = _yolo_like_model(tmp_path)
    with pytest.raises(ValueError, match="static"):
        quantize_model(model, tmp_path / "q.onnx", float_head=True)


def test_seeded_calibration_sample_is_reproducible_and_bounded(tmp_path):
    from ppe.tools.quantize import list_calibration_images

    for i in range(20):
        (tmp_path / f"{i:02d}.jpg").write_bytes(b"x")
    a = list_calibration_images(tmp_path, limit=5, seed=0)
    assert a == list_calibration_images(tmp_path, limit=5, seed=0) and len(a) == 5
    assert a != list_calibration_images(tmp_path, limit=5, seed=1)
    assert [p.name for p in list_calibration_images(tmp_path, limit=3)] == ["00.jpg", "01.jpg", "02.jpg"]
