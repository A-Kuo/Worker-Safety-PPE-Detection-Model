"""Generated models for tests and smoke checks (needs the ``onnx`` package: ``dev`` / ``quant`` extra).

``make_tiny_detector`` writes a few-kilobyte ONNX file with the same interface as an Ultralytics YOLO
export (static ``[1, 3, S, S]`` input, ``[1, 4 + nc, N]`` output, class names in the metadata) that
always "detects" the boxes you ask for. It lets CI exercise the real runtime — session creation,
preprocessing, decode, NMS, placement profiling — without third-party weights or torch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

# (class index, confidence, cx, cy, w, h) in input pixels
Box = tuple[int, float, float, float, float, float]

DEFAULT_NAMES = ("vest", "no_vest")
DEFAULT_BOXES: tuple[Box, ...] = ((0, 0.90, 20.0, 20.0, 16.0, 16.0), (1, 0.80, 44.0, 44.0, 12.0, 12.0))


def make_tiny_detector(
    path: str | Path,
    *,
    imgsz: int = 64,
    names: Sequence[str] = DEFAULT_NAMES,
    boxes: Sequence[Box] = DEFAULT_BOXES,
    anchors: int = 16,
) -> Path:
    """Write the model and return its path. Output is independent of the image content."""
    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:
        raise RuntimeError("make_tiny_detector needs the `onnx` package: pip install 'ppe[dev]'") from exc

    nc = len(names)
    if anchors <= 4 + nc:
        raise ValueError("anchors must exceed 4 + number of classes (the decoder relies on it)")
    if len(boxes) > anchors:
        raise ValueError("more boxes than anchors")
    table = np.zeros((1, 4 + nc, anchors), dtype=np.float32)
    for i, (cls, conf, cx, cy, w, h) in enumerate(boxes):
        table[0, :4, i] = (cx, cy, w, h)
        table[0, 4 + int(cls), i] = conf

    # out = table + mean(images) * 0 : real nodes that depend on the input (so placement profiling
    # has something to report) while the result stays the fixed table.
    nodes = [
        helper.make_node("ReduceMean", ["images"], ["mean"], name="/tiny/ReduceMean", keepdims=0),
        helper.make_node("Mul", ["mean", "zero"], ["nothing"], name="/tiny/Mul"),
        helper.make_node("Add", ["table", "nothing"], ["output0"], name="/tiny/Add"),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny_detector",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, imgsz, imgsz])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 4 + nc, anchors])],
        initializer=[
            numpy_helper.from_array(table, "table"),
            numpy_helper.from_array(np.zeros((), dtype=np.float32), "zero"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    meta = model.metadata_props.add()
    meta.key, meta.value = "names", repr(dict(enumerate(names)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path
