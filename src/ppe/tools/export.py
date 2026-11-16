"""Export a YOLO checkpoint to ONNX (static shapes by default) or OpenVINO IR."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


def _yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Exporting needs the training dependencies (torch + ultralytics), which the runtime "
            "install does not include. Install them with: pip install 'ppe[train]'"
        ) from exc
    return YOLO


def export_model(
    weights: Path,
    out: Path | None = None,
    *,
    imgsz: int = 640,
    dynamic: bool = False,
    simplify: bool = True,
    half: bool = False,
    opset: int = 17,
    fmt: str = "onnx",
) -> Path:
    """Export ``weights`` and return the written path.

    ``fmt="onnx"`` copies the export to ``out``. ``fmt="openvino"`` writes Ultralytics'
    ``*_openvino_model/`` directory next to the weights and ignores ``out``.
    """
    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    yolo = _yolo()
    if fmt == "openvino":
        model = yolo(str(weights))
        return Path(str(model.export(format="openvino", imgsz=imgsz, dynamic=dynamic, half=half)))
    if fmt != "onnx":
        raise ValueError(f"Unsupported export format: {fmt}")
    if out is None:
        raise ValueError("ONNX export needs an output path")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Ultralytics writes the .onnx next to the .pt it is given, silently overwriting any sibling
    # (this rewrote a tracked ONNX in the repo). Export from a scratch copy, then copy to ``out``.
    with tempfile.TemporaryDirectory(prefix="ppe_export_") as tmp:
        scratch = Path(tmp) / weights.name
        shutil.copyfile(weights, scratch)
        exported = Path(
            str(
                yolo(str(scratch)).export(
                    format="onnx", imgsz=imgsz, dynamic=dynamic, simplify=simplify, half=half, opset=opset
                )
            )
        )
        shutil.copyfile(exported, out)
    return out
