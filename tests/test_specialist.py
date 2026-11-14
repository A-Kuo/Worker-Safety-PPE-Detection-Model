"""Class-pair specialists replace only their owned classes; vest and helmet can both attach."""

from __future__ import annotations

import numpy as np

from ppe.compliance import Detection
from ppe.inference import _merge_specialist_rows
from ppe.runtime.backends import _onnx_metadata_names
from ppe.runtime.config import ExecutionPolicy
from ppe.runtime.session import EdgeSession, SpecialistSlot
from ppe.specialist import merge_detections, replace_classes_for


def _det(name: str, conf: float, x: float = 0.0) -> Detection:
    return Detection(cls_name=name, conf=conf, xyxy=(x, 0.0, x + 10.0, 10.0))


def test_merge_replaces_only_specialist_classes():
    main = [_det("helmet", 0.9), _det("vest", 0.3), _det("no_vest", 0.2), _det("person", 0.8)]
    spec = [_det("vest", 0.7), _det("no_vest", 0.6), _det("helmet", 0.99)]  # helmet must be ignored
    merged = merge_detections(main, spec)
    assert sorted((d.cls_name, d.conf) for d in merged) == [
        ("helmet", 0.9), ("no_vest", 0.6), ("person", 0.8), ("vest", 0.7),
    ]


def test_merge_with_no_specialist_detections_drops_main_vest_boxes():
    # The specialist is authoritative for vest status: silence from it means no vest/no_vest box.
    merged = merge_detections([_det("vest", 0.9), _det("helmet", 0.9)], [])
    assert [d.cls_name for d in merged] == ["helmet"]


def test_replace_classes_for_known_pairs():
    assert replace_classes_for(["vest", "no_vest"]) == frozenset({"vest", "no_vest"})
    assert replace_classes_for(["helmet", "no_helmet"]) == frozenset({"helmet", "no_helmet"})
    # Partial name still selects the full known pair so empty predictions still clear the right classes.
    assert replace_classes_for(["no_helmet"]) == frozenset({"helmet", "no_helmet"})


def test_helmet_specialist_does_not_touch_vest():
    main = [_det("helmet", 0.4), _det("no_helmet", 0.3), _det("vest", 0.9), _det("person", 0.8)]
    helmet = [_det("helmet", 0.95), _det("no_helmet", 0.05)]
    merged = merge_detections(main, helmet, replace=replace_classes_for(["helmet", "no_helmet"]))
    assert sorted((d.cls_name, d.conf) for d in merged) == [
        ("helmet", 0.95), ("no_helmet", 0.05), ("person", 0.8), ("vest", 0.9),
    ]


def test_sequential_vest_then_helmet_specialists():
    main = [_det("helmet", 0.4), _det("vest", 0.3), _det("person", 0.8)]
    after_vest = merge_detections(main, [_det("no_vest", 0.7)], replace=frozenset({"vest", "no_vest"}))
    after_both = merge_detections(
        after_vest, [_det("helmet", 0.91)], replace=frozenset({"helmet", "no_helmet"})
    )
    assert sorted((d.cls_name, d.conf) for d in after_both) == [
        ("helmet", 0.91), ("no_vest", 0.7), ("person", 0.8),
    ]


class _Backend:
    name = "fake"

    def __init__(self, dets, names=None):
        self.dets, self.asked = dets, None
        self.class_names = names or {}

    def predict(self, image_bgr, *, conf):
        self.asked = conf
        return [d for d in self.dets if d.conf >= conf]

    def info(self):
        return {"class_names": dict(self.class_names)}


def test_edge_session_merges_specialist_and_applies_class_thresholds(tmp_path):
    main = _Backend([_det("helmet", 0.9), _det("vest", 0.6)])
    spec = _Backend([_det("no_vest", 0.08), _det("vest", 0.7)], {0: "vest", 1: "no_vest"})
    policy = ExecutionPolicy(conf=0.25, class_conf={"no_vest": 0.05})
    session = EdgeSession(
        backend=main,
        policy=policy,
        model_path=tmp_path / "m.onnx",
        specialist_backends=[SpecialistSlot(backend=spec, replace=frozenset({"vest", "no_vest"}), path=tmp_path / "v.onnx")],
        specialist_backend=spec,
    )
    kept = session.predict(np.zeros((8, 8, 3), np.uint8))
    assert main.asked == spec.asked == 0.05  # both models are asked for the lowest threshold in play
    assert sorted((d.cls_name, d.conf) for d in kept) == [("helmet", 0.9), ("no_vest", 0.08), ("vest", 0.7)]


def test_edge_session_merges_vest_and_helmet_specialists_on_npu_shaped_path(tmp_path):
    """NPU path: two ONNX specialists (disjoint pairs) both replace their classes."""
    main = _Backend([_det("helmet", 0.4), _det("no_helmet", 0.3), _det("vest", 0.3), _det("person", 0.9)])
    vest = _Backend([_det("no_vest", 0.8)], {0: "vest", 1: "no_vest"})
    helmet = _Backend([_det("helmet", 0.92)], {0: "helmet", 1: "no_helmet"})
    policy = ExecutionPolicy(conf=0.25, prefer_onnx=True, npu_only=True, providers=("openvino", "cpu"))
    session = EdgeSession(
        backend=main,
        policy=policy,
        model_path=tmp_path / "main.onnx",
        specialist_backends=[
            SpecialistSlot(backend=vest, replace=frozenset({"vest", "no_vest"}), path=tmp_path / "vest.onnx"),
            SpecialistSlot(backend=helmet, replace=frozenset({"helmet", "no_helmet"}), path=tmp_path / "helmet.onnx"),
        ],
    )
    kept = session.predict(np.zeros((8, 8, 3), np.uint8))
    assert sorted((d.cls_name, d.conf) for d in kept) == [
        ("helmet", 0.92), ("no_vest", 0.8), ("person", 0.9),
    ]


def test_policy_reads_specialist_from_env(monkeypatch):
    monkeypatch.setenv("PPE_SPECIALIST", "models/vest_specialist.onnx")
    monkeypatch.delenv("PPE_SPECIALISTS", raising=False)
    policy = ExecutionPolicy.from_env()
    assert policy.specialist == "models/vest_specialist.onnx"
    assert policy.specialists == ("models/vest_specialist.onnx",)


def test_policy_reads_multiple_specialists_from_env(monkeypatch):
    monkeypatch.setenv(
        "PPE_SPECIALISTS",
        "models/vest_specialist.onnx, models/helmet_specialist.onnx",
    )
    monkeypatch.setenv("PPE_SPECIALIST", "models/vest_specialist.onnx")  # de-duped
    policy = ExecutionPolicy.from_env()
    assert policy.specialists == (
        "models/vest_specialist.onnx",
        "models/helmet_specialist.onnx",
    )


class _Boxes:
    def __init__(self, rows):
        self.data = np.asarray(rows, dtype=np.float32).reshape(-1, 6)

    def __len__(self):
        return len(self.data)


class _Result:
    def __init__(self, names, rows):
        self.names, self.boxes = names, _Boxes(rows)


def test_row_merge_maps_specialist_classes_onto_the_main_models_ids():
    # Main model in raw Hexmon-style naming: ids are NOT unified order, so mapping must go by name.
    main = _Result({0: "Hardhat", 1: "Safety Vest", 2: "NO-Safety Vest"}, [
        [0, 0, 5, 5, 0.9, 0],      # Hardhat      -> kept
        [0, 0, 5, 5, 0.4, 1],      # Safety Vest  -> dropped (specialist is authoritative)
    ])
    spec = _Result({0: "vest", 1: "no_vest"}, [
        [1, 1, 6, 6, 0.7, 0],      # vest    -> main id 1
        [2, 2, 7, 7, 0.06, 1],     # no_vest -> main id 2
    ])
    rows = _merge_specialist_rows(main, spec)
    assert sorted(map(tuple, rows[:, [4, 5]].tolist())) == [
        (np.float32(0.06).item(), 2.0), (np.float32(0.7).item(), 1.0), (np.float32(0.9).item(), 0.0),
    ]


def test_row_merge_helmet_specialist_maps_onto_hardhat_ids():
    main = _Result({0: "Hardhat", 1: "NO-Hardhat", 2: "Safety Vest"}, [
        [0, 0, 5, 5, 0.4, 0],
        [0, 0, 5, 5, 0.9, 2],
    ])
    spec = _Result({0: "helmet", 1: "no_helmet"}, [
        [1, 1, 6, 6, 0.88, 0],
    ])
    rows = _merge_specialist_rows(main, spec)
    assert sorted(map(tuple, rows[:, [4, 5]].tolist())) == [
        (np.float32(0.88).item(), 0.0), (np.float32(0.9).item(), 2.0),
    ]


def test_row_merge_is_a_noop_if_the_main_model_has_no_vest_classes():
    assert _merge_specialist_rows(_Result({0: "cone"}, []), _Result({0: "vest"}, [[0, 0, 1, 1, 0.9, 0]])) is None


class _Meta:
    def __init__(self, names):
        self.custom_metadata_map = {"names": names} if names is not None else {}


class _MetaSession:
    def __init__(self, names):
        self._names = names

    def get_modelmeta(self):
        return _Meta(self._names)


def test_onnx_metadata_names_are_read_for_a_two_class_model():
    assert _onnx_metadata_names(_MetaSession("{0: 'vest', 1: 'no_vest'}")) == {0: "vest", 1: "no_vest"}


def test_onnx_metadata_names_are_read_for_helmet_specialist():
    assert _onnx_metadata_names(_MetaSession("{0: 'helmet', 1: 'no_helmet'}")) == {
        0: "helmet",
        1: "no_helmet",
    }


def test_onnx_metadata_names_fall_back_to_none():
    assert _onnx_metadata_names(_MetaSession(None)) is None
    assert _onnx_metadata_names(object()) is None  # session without metadata support
    assert _onnx_metadata_names(_MetaSession("not a dict")) is None
