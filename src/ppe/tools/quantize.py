"""INT8-quantize an ONNX model (dynamic by default, static with calibration images)."""

from __future__ import annotations

from pathlib import Path

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


def list_calibration_images(folder: Path, limit: int = 100) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Calibration folder not found: {folder}")
    images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())
    if not images:
        raise FileNotFoundError(f"No images under {folder}")
    return images[:limit]


def quantize_dynamic(model: Path, out: Path) -> Path:
    q = _quantization()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    q.quantize_dynamic(model_input=str(model), model_output=str(out), weight_type=q.QuantType.QInt8)
    return out


def quantize_static(model: Path, out: Path, calib_dir: Path, imgsz: int = 640) -> Path:
    q = _quantization()
    import cv2
    import onnxruntime as ort

    from ppe.runtime.backends import _letterbox

    images = list_calibration_images(calib_dir)
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
    q.quantize_static(
        model_input=str(model),
        model_output=str(out),
        calibration_data_reader=_Reader(),
        weight_type=q.QuantType.QInt8,
        activation_type=q.QuantType.QInt8,
    )
    return out


def quantize_model(
    model: Path,
    out: Path | None = None,
    *,
    calibration: Path | None = None,
    static: bool = False,
    imgsz: int = 640,
) -> Path:
    model = Path(model)
    if not model.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model}")
    if static and calibration is None:
        raise ValueError("Static quantization requires a calibration image folder")
    out = Path(out) if out else default_output(model)
    if static:
        return quantize_static(model, out, Path(calibration), imgsz)
    return quantize_dynamic(model, out)
