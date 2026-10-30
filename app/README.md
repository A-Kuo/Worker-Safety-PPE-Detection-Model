# PPE detection demo (API + UI)

Local rewrite of the deploy step: FastAPI for inference, Streamlit for a compliance review UI. Both go through `app.runtime.DemoEngine`, which prefers the **ONNX / NPU edge path** and falls back to Torch only for `.pt` weights.

```text
from app.runtime import build_engine
from ppe.compliance import WorkerCompliance, Detection, associate_ppe_to_persons
```

Core Ultralytics / torch / onnxruntime deps live in the **repo-root** `requirements.txt`. This folder only adds the demo extras. Edge / NPU details: [`docs/edge_npu.md`](../docs/edge_npu.md).

## Install

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -r app/requirements.txt
```

`app/requirements.txt`: `fastapi`, `uvicorn`, `python-multipart`, `streamlit`, `opencv-python-headless`.

For Intel AI Boost NPU (OpenVINO EP), use a dedicated env — see `docs/edge_npu.md` (`pip install -e ".[edge-intel-npu]"`).

## Weights

The service reads `PPE_WEIGHTS` if set. Otherwise it uses the first file that exists (**ONNX first**):

1. `models/best.int8.onnx`
2. `models/best.onnx`
3. `baselines/.../best.int8.onnx` / `best.onnx`
4. `baselines/snehilsanyal_yolov8n_css/models/best.pt`
5. `models/best.pt`

Export ONNX for timely NPU camera inference:

```bash
python scripts/export_onnx.py --weights baselines/snehilsanyal_yolov8n_css/models/best.pt --out models/best.onnx
```

Optional env: `PPE_DEVICE` (Torch fallback), `PPE_PROVIDERS`, `PPE_NPU_ONLY`, `PPE_OPENVINO_DEVICE` (default demo bias `AUTO:NPU,GPU`), `PPE_PREFER_NPU` (default true), `PPE_API_HOST` / `PPE_API_PORT`, `PPE_API_URL` (UI live-stream base).

## Launch the API

```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

Or: `python -m app.api.main`

OpenAPI: http://127.0.0.1:8000/docs

### Endpoints

| Method | Path | What it does |
|---|---|---|
| `GET` | `/health` | Weights, ready flag, engine kind, resolved ORT providers, OpenVINO device |
| `POST` | `/predict/image` | Multipart image → JSON detections + compliance labels. `?return_image=true` adds a base64 JPEG |
| `POST` | `/predict/video` | Multipart video, strided/capped (default 300 frames / 45s). Summary JSON. `?return_clip=true` writes `output/app/*.mp4` and returns `annotated_path` + `/clips/{name}` |
| `GET` | `/clips/{name}` | Download a clip produced by `/predict/video` |
| `GET` | `/stream` | MJPEG camera path. NPU-first engine; `timely=true` (default) drains the capture buffer so inference stays on the newest frame |

Image query extras: `conf` (default `0.25`), `return_image`.  
Video query extras: `conf`, `return_clip`, `max_frames`, `max_seconds`.  
Stream query extras: `source`, `conf`, `max_fps` (default 12), `timely` (default true).

CORS is open (`*`) for a local demo.

### No camera → 503

`GET /stream` opens the requested source immediately. If that fails (no webcam, bad RTSP, missing file), the API returns **HTTP 503** with a short hint. Headless hosts without a camera will always 503 on numeric webcam indexes.

## Launch the UI

In a second terminal, from the repository root:

```bash
streamlit run app/ui/streamlit_app.py
```

Camera-first layout: live API MJPEG embed, snapshot capture with latency readout, then Image / Video tabs. Sidebar exposes NPU preference, OpenVINO `device_type`, provider pins, and the API base URL.

Continuous live video is the API MJPEG route (`/stream?timely=true`), not a Socket.IO dashboard.

## curl examples

```bash
curl http://127.0.0.1:8000/health

curl -F "file=@frame.jpg" "http://127.0.0.1:8000/predict/image?return_image=true"

curl -F "file=@clip.mp4" "http://127.0.0.1:8000/predict/video?return_clip=true&max_frames=120"
```
