"""Strict-NPU mode, placement verification and the runtime fingerprint (no NPU hardware needed)."""

from __future__ import annotations

import json

import pytest

from ppe.runtime.backends import is_static_shape, make_session_options, plan_providers
from ppe.runtime.config import ExecutionPolicy
from ppe.runtime.placement import parse_profile, verify_placement
from ppe.runtime.providers import provider_options_for, resolve_providers

OV = "OpenVINOExecutionProvider"
CPU = "CPUExecutionProvider"


def _ev(node, op, provider, suffix="_kernel_time"):
    return {"cat": "Node", "name": node + suffix, "args": {"op_name": op, "provider": provider}}


# --- profile parsing ---------------------------------------------------------------------------

def test_parse_profile_counts_each_node_once_and_flags_cpu_ops():
    events = [
        {"cat": "Session", "name": "model_run"},  # ignored: not a node event
        _ev("conv1", "Conv", OV),
        _ev("conv1", "Conv", OV, suffix="_fence_before"),  # fence events must not double count
        _ev("sig", "Sigmoid", OV),
        _ev("post", "TopK", CPU),
        _ev("copy", "MemcpyToHost", CPU),  # data movement: reported, never CPU compute
    ]
    out = parse_profile(events)
    assert out["nodes_total"] == 3
    assert out["by_provider"] == {OV: 2, CPU: 1}
    assert out["npu_nodes"] == 2
    assert out["npu_fraction"] == pytest.approx(0.6667, abs=1e-4)
    assert out["cpu_resident"] == [{"node": "post", "op": "TopK"}]
    assert out["transfer_nodes"] == 1


def test_parse_profile_all_npu_has_no_cpu_resident():
    out = parse_profile([_ev("a", "Conv", OV), _ev("b", "Mul", OV)])
    assert out["cpu_resident"] == []
    assert out["npu_fraction"] == 1.0


def test_apply_device_pin_never_overclaims_npu_for_gpu_or_cpu_devices():
    from ppe.runtime.placement import apply_device_pin

    base = parse_profile([_ev("a", "Conv", OV), _ev("b", "Mul", OV)])
    unpinned = apply_device_pin(base, False)  # OpenVINO on GPU/CPU/AUTO
    assert unpinned["npu_nodes"] == 0 and unpinned["npu_fraction"] == 0.0
    assert unpinned["accelerator_nodes"] == 2
    pinned = apply_device_pin(base, True)
    assert pinned["npu_nodes"] == 2 and pinned["npu_fraction"] == 1.0
    assert apply_device_pin(base, None)["npu_nodes"] == 2  # non-OpenVINO EPs (QNN etc.): unchanged


def test_parse_profile_empty():
    out = parse_profile([])
    assert out["nodes_total"] == 0
    assert out["npu_fraction"] is None


def test_parse_profile_falls_back_to_non_kernel_events():
    events = [_ev("a", "Conv", OV, suffix=""), _ev("b", "Mul", CPU, suffix="")]
    out = parse_profile(events)
    assert out["by_provider"] == {OV: 1, CPU: 1}


# --- strict policy -----------------------------------------------------------------------------

def test_strict_implies_npu_only_no_cpu_and_npu_device():
    p = ExecutionPolicy(strict_npu=True)
    assert p.effective_npu_only is True
    assert p.effective_allow_cpu_fallback is False
    assert p.effective_openvino_device == "NPU"
    # An explicit device is respected (the user may knowingly pick e.g. GPU for comparison).
    assert ExecutionPolicy(strict_npu=True, openvino_device_type="GPU").effective_openvino_device == "GPU"


def test_non_strict_defaults_unchanged():
    p = ExecutionPolicy()
    assert p.effective_npu_only is False
    assert p.effective_allow_cpu_fallback is True
    assert p.effective_openvino_device is None


def test_strict_from_env(monkeypatch):
    monkeypatch.setenv("PPE_STRICT_NPU", "1")
    monkeypatch.setenv("PPE_OPENVINO_CACHE_DIR", "cache")
    p = ExecutionPolicy.from_env()
    assert p.strict_npu and p.openvino_cache_dir == "cache"


def test_exclude_cpu_strips_explicitly_pinned_cpu(monkeypatch):
    monkeypatch.setattr("ppe.runtime.providers.available_ort_names", lambda: [OV, CPU])
    assert resolve_providers(["openvino", "cpu"], npu_only=True) == [OV, CPU]
    assert resolve_providers(["openvino", "cpu"], npu_only=True, exclude_cpu=True) == [OV]


def test_plan_providers_strict_never_includes_cpu(monkeypatch):
    monkeypatch.setattr("ppe.runtime.providers.available_ort_names", lambda: [OV, CPU])
    providers, options = plan_providers(ExecutionPolicy(strict_npu=True, providers=("openvino", "cpu")))
    assert providers == [OV]
    assert options == [{"device_type": "NPU"}]


def test_plan_providers_strict_refuses_without_npu_ep(monkeypatch):
    monkeypatch.setattr("ppe.runtime.providers.available_ort_names", lambda: [CPU])
    with pytest.raises(RuntimeError, match="NPU execution required"):
        plan_providers(ExecutionPolicy(strict_npu=True))


def test_provider_options_cache_dir_and_qnn_backend():
    assert provider_options_for("openvino", openvino_device_type="NPU", openvino_cache_dir="c") == {
        "device_type": "NPU",
        "cache_dir": "c",
    }
    assert provider_options_for("qnn", qnn_backend_path="X.dll") == {"backend_path": "X.dll"}
    assert "backend_path" in provider_options_for("qnn")
    assert provider_options_for("cpu") == {}


# --- session options ---------------------------------------------------------------------------

class _FakeSessionOptions:
    def __init__(self):
        self.config = {}
        self.enable_profiling = False
        self.profile_file_prefix = None

    def add_session_config_entry(self, key, value):
        self.config[key] = value


class _FakeOrt:
    SessionOptions = _FakeSessionOptions


def test_strict_session_options_disable_cpu_ep_fallback():
    so = make_session_options(_FakeOrt, ExecutionPolicy(strict_npu=True))
    assert so.config == {"session.disable_cpu_ep_fallback": "1"}
    assert make_session_options(_FakeOrt, ExecutionPolicy()).config == {}


def test_profile_prefix_enables_profiling():
    so = make_session_options(_FakeOrt, ExecutionPolicy(), profile_prefix="p")
    assert so.enable_profiling is True and so.profile_file_prefix == "p"


def test_is_static_shape():
    assert is_static_shape([1, 3, 640, 640])
    assert not is_static_shape(["batch", 3, 640, 640])
    assert not is_static_shape([1, 3, None, 640])


# --- backend strict behaviour (fake ORT session) -----------------------------------------------

class _Inp:
    def __init__(self, shape):
        self.name = "images"
        self.shape = shape


def _fake_session_cls(shape, fail=False):
    class _S:
        def __init__(self, model_path, providers=None, provider_options=None, sess_options=None):
            if fail:
                raise RuntimeError("Could not find an implementation for NodeX")
            self.providers = providers
            self.sess_options = sess_options

        def get_inputs(self):
            return [_Inp(shape)]

        def get_providers(self):
            return self.providers

    return _S


def _model(tmp_path):
    p = tmp_path / "m.onnx"
    p.write_bytes(b"x")
    return p


def test_backend_strict_rejects_dynamic_shape(monkeypatch, tmp_path):
    import onnxruntime as ort

    from ppe.runtime.backends import OnnxRuntimeBackend

    monkeypatch.setattr(ort, "InferenceSession", _fake_session_cls(["batch", 3, 640, 640]))
    monkeypatch.setattr("ppe.runtime.providers.available_ort_names", lambda: [OV, CPU])
    with pytest.raises(RuntimeError, match="static input shape"):
        OnnxRuntimeBackend(_model(tmp_path), ExecutionPolicy(strict_npu=True, providers=("openvino",)))


def test_backend_strict_wraps_cpu_fallback_refusal(monkeypatch, tmp_path):
    import onnxruntime as ort

    from ppe.runtime.backends import OnnxRuntimeBackend

    monkeypatch.setattr(ort, "InferenceSession", _fake_session_cls([1, 3, 640, 640], fail=True))
    monkeypatch.setattr("ppe.runtime.providers.available_ort_names", lambda: [OV, CPU])
    with pytest.raises(RuntimeError, match="Strict NPU mode.*could not be placed entirely"):
        OnnxRuntimeBackend(_model(tmp_path), ExecutionPolicy(strict_npu=True, providers=("openvino",)))


def test_backend_non_strict_does_not_wrap_errors(monkeypatch, tmp_path):
    import onnxruntime as ort

    from ppe.runtime.backends import OnnxRuntimeBackend

    monkeypatch.setattr(ort, "InferenceSession", _fake_session_cls([1, 3, 640, 640], fail=True))
    monkeypatch.setattr("ppe.runtime.providers.available_ort_names", lambda: [CPU])
    with pytest.raises(RuntimeError, match="NodeX"):
        OnnxRuntimeBackend(_model(tmp_path), ExecutionPolicy())


def test_backend_info_reports_strict_and_fingerprint(monkeypatch, tmp_path):
    import onnxruntime as ort

    from ppe.runtime.backends import OnnxRuntimeBackend

    monkeypatch.setattr(ort, "InferenceSession", _fake_session_cls([1, 3, 640, 640]))
    monkeypatch.setattr("ppe.runtime.providers.available_ort_names", lambda: [OV, CPU])
    be = OnnxRuntimeBackend(_model(tmp_path), ExecutionPolicy(strict_npu=True, providers=("openvino",)))
    info = be.info()
    assert info["strict_npu"] is True and info["npu_only"] is True
    assert info["static_input"] is True
    assert info["providers_active"] == [OV]
    assert info["openvino_device_type"] == "NPU"
    fp = info["fingerprint"]
    assert fp["model"]["sha256"] and fp["model"]["bytes"] == 1
    assert "onnxruntime" in fp and "python" in fp


def test_openvino_device_is_npu():
    from ppe.runtime.backends import openvino_device_is_npu

    assert openvino_device_is_npu("NPU") and openvino_device_is_npu("npu") and openvino_device_is_npu("NPU.0")
    for bad in (None, "", "CPU", "GPU", "AUTO:NPU,GPU", "HETERO:NPU,CPU", "NPU,GPU", "NPU.x"):
        assert not openvino_device_is_npu(bad), bad


@pytest.mark.parametrize("device", ["GPU", "CPU", "AUTO:NPU,GPU"])
def test_strict_refuses_non_npu_openvino_device(monkeypatch, device):
    monkeypatch.setattr("ppe.runtime.providers.available_ort_names", lambda: [OV, CPU])
    policy = ExecutionPolicy(strict_npu=True, providers=("openvino",), openvino_device_type=device)
    with pytest.raises(RuntimeError, match="pinned to the NPU"):
        plan_providers(policy)


def test_non_strict_allows_other_openvino_devices(monkeypatch):
    monkeypatch.setattr("ppe.runtime.providers.available_ort_names", lambda: [OV, CPU])
    providers, options = plan_providers(ExecutionPolicy(providers=("openvino",), openvino_device_type="GPU"))
    assert providers == [OV, CPU] or providers == [OV]
    assert options[0] == {"device_type": "GPU"}


def _session_with_active(active):
    class _S:
        def __init__(self, model_path, providers=None, provider_options=None, sess_options=None):
            pass

        def get_inputs(self):
            return [_Inp([1, 3, 640, 640])]

        def get_providers(self):
            return list(active)

    return _S


def test_requested_ep_that_failed_to_load_is_an_error_in_strict_mode(monkeypatch, tmp_path):
    import onnxruntime as ort

    from ppe.runtime.backends import OnnxRuntimeBackend

    # ORT only *warns* when e.g. openvino.dll is missing, then runs on the CPU. Never accept that.
    monkeypatch.setattr(ort, "InferenceSession", _session_with_active([CPU]))
    monkeypatch.setattr("ppe.runtime.providers.available_ort_names", lambda: [OV, CPU])
    with pytest.raises(RuntimeError, match="did not load"):
        OnnxRuntimeBackend(_model(tmp_path), ExecutionPolicy(strict_npu=True, providers=("openvino",)))


def test_requested_ep_that_failed_to_load_warns_when_not_strict(monkeypatch, tmp_path):
    import onnxruntime as ort

    from ppe.runtime.backends import OnnxRuntimeBackend

    monkeypatch.setattr(ort, "InferenceSession", _session_with_active([CPU]))
    monkeypatch.setattr("ppe.runtime.providers.available_ort_names", lambda: [OV, CPU])
    with pytest.warns(RuntimeWarning, match="did not load"):
        be = OnnxRuntimeBackend(_model(tmp_path), ExecutionPolicy(providers=("openvino", "cpu")))
    assert be.info()["providers_missing"] == [OV]


# --- real ONNX Runtime on CPU: the verifier pipeline end to end --------------------------------

def _tiny_model(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])
    y = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3, 32, 32])
    graph = helper.make_graph([helper.make_node("Relu", ["images"], ["out"], name="relu0")], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "tiny.onnx"
    onnx.save(model, str(path))
    return path


def test_verify_placement_soft_mode_reports_cpu_resident_ops(tmp_path):
    pytest.importorskip("onnxruntime")
    path = _tiny_model(tmp_path)
    report = verify_placement(path, ExecutionPolicy(imgsz=32), runs=2)
    json.dumps(report)  # must be serializable for --out
    assert report["strict_npu"] is False
    assert report["method"] == "profile"
    assert report["nodes_total"] >= 1
    assert report["by_provider"].get(CPU, 0) >= 1
    assert report["ok"] is False  # a CPU-only box must NOT be reported as NPU-resident
    assert any(n["op"] == "Relu" for n in report["cpu_resident"])


def test_verify_placement_strict_refuses_on_cpu_only_box(tmp_path, monkeypatch):
    pytest.importorskip("onnxruntime")
    monkeypatch.setattr("ppe.runtime.providers.available_ort_names", lambda: [CPU])
    with pytest.raises(RuntimeError, match="NPU execution required"):
        verify_placement(_tiny_model(tmp_path), ExecutionPolicy(strict_npu=True))


def test_cli_verify_npu_exits_nonzero_with_json_error_when_no_npu(tmp_path, monkeypatch, capsys):
    pytest.importorskip("onnxruntime")
    from ppe import cli

    monkeypatch.setattr("ppe.runtime.providers.available_ort_names", lambda: [CPU])
    code = cli.main(["verify-npu", "--weights", str(_tiny_model(tmp_path))])
    out = json.loads(capsys.readouterr().out)
    assert code == 1
    assert out["ok"] is False and out["strict_npu"] is True
    assert "NPU execution required" in out["error"]


def test_cli_verify_npu_soft_exits_nonzero_on_cpu_and_reports_ops(tmp_path, capsys):
    pytest.importorskip("onnxruntime")
    from ppe import cli

    code = cli.main(["verify-npu", "--soft", "--imgsz", "32", "--weights", str(_tiny_model(tmp_path))])
    out = json.loads(capsys.readouterr().out)
    assert code == 1  # CPU-resident ops => not NPU-proven
    assert out["cpu_resident"]
