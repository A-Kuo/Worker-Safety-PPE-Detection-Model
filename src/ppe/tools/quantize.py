"""INT8-quantize an ONNX model (dynamic by default, static with calibration images)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _quantization():
    try:
        import onnx  # noqa: F401 - onnxruntime.quantization needs it
        from onnxruntime import quantization
    except ImportError as exc:
        raise RuntimeError(
            "Quantizing needs ONNX Runtime plus the `onnx` package. Install them with: "
            "pip install 'ppe[cpu,quant]'"
        ) from exc
    return quantization


def default_output(model: Path) -> Path:
    """``<stem>.int8.onnx`` next to the input model."""
    return Path(model).with_name(Path(model).stem + ".int8.onnx")


def list_calibration_images(folder: Path, limit: int = 100, seed: int | None = None) -> list[Path]:
    """Calibration images under ``folder``. With ``seed`` the sample is random but reproducible;
    without it the first ``limit`` by name are used (which can be one scene repeated)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Calibration folder not found: {folder}")
    images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())
    if not images:
        raise FileNotFoundError(f"No images under {folder}")
    if seed is not None and len(images) > limit:
        import random

        images = sorted(random.Random(seed).sample(images, limit))
    return images[:limit]


def yolo_head_float_nodes(model: Path) -> list[str]:
    """Node names of an Ultralytics YOLO detection head that must stay in float.

    The head concatenates box coordinates (0..imgsz) with class scores (0..1) into one output tensor.
    Quantized with a single scale, the scores collapse to zero: measured AP50 0.0 on the CPU EP for
    the vest model, while the NPU happened to survive it. Keeping the head's non-convolution ops (DFL,
    distance-to-box arithmetic, sigmoid, final concat) in float fixes it on every device.

    The head is the highest-numbered ``/model.<n>/`` module. Returns [] for other graph layouts.
    """
    import re

    import onnx

    graph = onnx.load(str(model), load_external_data=False).graph
    pat = re.compile(r"^/model\.(\d+)/")
    indices = [int(m.group(1)) for n in graph.node if (m := pat.match(n.name))]
    if not indices:
        return []
    prefix = f"/model.{max(indices)}/"
    return [
        n.name
        for n in graph.node
        if n.name.startswith(prefix) and (n.op_type != "Conv" or "/dfl/" in n.name)
    ]


def quantize_dynamic(model: Path, out: Path) -> Path:
    q = _quantization()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    q.quantize_dynamic(model_input=str(model), model_output=str(out), weight_type=q.QuantType.QInt8)
    return out


def quantize_static(
    model: Path,
    out: Path,
    calib_dir: Path,
    imgsz: int = 640,
    *,
    qdq: bool = False,
    per_channel: bool = False,
    limit: int = 100,
    seed: int | None = None,
    exclude_nodes: Sequence[str] = (),
) -> Path:
    """Static INT8. ``qdq=True`` writes QDQ format with uint8 activations (what NPU toolchains expect);
    the default keeps the original QOperator / int8-activation behaviour."""
    q = _quantization()
    import cv2
    import onnxruntime as ort

    from ppe.runtime.backends import _letterbox

    images = list_calibration_images(calib_dir, limit=limit, seed=seed)
    input_name = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"]).get_inputs()[0].name

    class _Reader(q.CalibrationDataReader):
        def __init__(self) -> None:
            self._it = iter(images)

        def get_next(self):
            for path in self._it:
                bgr = cv2.imread(str(path))
                if bgr is None:
                    continue
                # Same preprocessing as inference (fresh array each call: the calibrator keeps it).
                tensor, _ = _letterbox(bgr, imgsz)
                return {input_name: tensor}
            return None

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {}
    if qdq:
        kwargs.update(quant_format=q.QuantFormat.QDQ, activation_type=q.QuantType.QUInt8)
    else:
        kwargs.update(activation_type=q.QuantType.QInt8)
    if exclude_nodes:
        kwargs["nodes_to_exclude"] = list(exclude_nodes)
    q.quantize_static(
        model_input=str(model),
        model_output=str(out),
        calibration_data_reader=_Reader(),
        weight_type=q.QuantType.QInt8,
        per_channel=per_channel,
        **kwargs,
    )
    return out


def quantize_model(
    model: Path,
    out: Path | None = None,
    *,
    calibration: Path | None = None,
    static: bool = False,
    imgsz: int = 640,
    qdq: bool = False,
    per_channel: bool = False,
    limit: int = 100,
    seed: int | None = None,
    exclude_nodes: Sequence[str] = (),
    float_head: bool = False,
) -> Path:
    model = Path(model)
    if not model.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model}")
    if static and calibration is None:
        raise ValueError("Static quantization requires a calibration image folder")
    out = Path(out) if out else default_output(model)
    if float_head:
        if not static:
            raise ValueError("--float-head applies to static quantization")
        head = yolo_head_float_nodes(model)
        if not head:
            raise ValueError(f"No Ultralytics-style detection head found in {model.name}")
        exclude_nodes = [*exclude_nodes, *head]
    if static:
        return quantize_static(
            model, out, Path(calibration), imgsz, qdq=qdq, per_channel=per_channel, limit=limit,
            seed=seed, exclude_nodes=exclude_nodes,
        )
    return quantize_dynamic(model, out)
