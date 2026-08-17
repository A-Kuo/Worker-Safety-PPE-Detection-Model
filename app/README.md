# PPE detection demo (API + UI)

Local rewrite of the deploy step: FastAPI for inference, Streamlit for a compliance review UI. Both call `ppe` (src layout; `src.ppe` fallback) — they do not reimplement detection or person–PPE association.

```text
from ppe.inference import PPEDetector
from ppe.compliance import WorkerCompliance, Detection, associate_ppe_to_persons
```

Core Ultralytics / torch deps live in the **repo-root** `requirements.txt`. This folder only adds the demo extras.

## Install

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -r app/requirements.txt
```

`app/requirements.txt`: `fastapi`, `uvicorn`, `python-multipart`, `streamlit`, `opencv-python-headless`.

## Weights

The service reads `PPE_WEIGHTS` if set. Otherwise it uses the first file that exists:

1. `baselines/snehilsanyal_yolov8n_css/models/best.pt`
2. `models/best.pt`

Optional: `PPE_DEVICE` (`cpu`, `cuda`, `0`, …), `PPE_API_HOST` (default `0.0.0.0`), `PPE_API_PORT` (default `8000`).

PowerShell:

```powershell
$env:PPE_WEIGHTS = "baselines/snehilsanyal_yolov8n_css/models/best.pt"
```

## Launch the API

```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

Or: `python -m app.api.main`

OpenAPI: http://127.0.0.1:8000/docs

### Endpoints

| Method | Path | What it does |
|---|---|---|
| `GET` | `/health` | Weights path, whether `ppe` imported, ready flag |
| `POST` | `/predict/image` | Multipart image → JSON detections + compliance labels. `?return_image=true` adds a base64 JPEG |
| `POST` | `/predict/video` | Multipart video, strided/capped (default 300 frames / 45s). Summary JSON. `?return_clip=true` writes `output/app/*.mp4` and returns `annotated_path` + `/clips/{name}` |
| `GET` | `/clips/{name}` | Download a clip produced by `/predict/video` |
| `GET` | `/stream` | MJPEG. `?source=0` (default webcam), `?source=rtsp://…`, or `?source=C:/path/video.mp4` |

Image query extras: `conf` (default `0.25`), `return_image`.  
Video query extras: `conf`, `return_clip`, `max_frames`, `max_seconds`.  
Stream query extras: `source`, `conf`, `max_fps` (default 12).

CORS is open (`*`) for a local demo.

### No camera → 503

`GET /stream` opens the requested source immediately. If that fails (no webcam, bad RTSP, missing file), the API returns **HTTP 503** with a short hint. Headless hosts without a camera will always 503 on numeric webcam indexes.

## Launch the UI

In a second terminal, from the repository root:

```bash
streamlit run app/ui/streamlit_app.py
```

Sidebar: weights path and confidence. Tabs: **Image**, **Video**, **Webcam** (`st.camera_input` snapshot). Each run shows the annotated frame plus compliance strings from `WorkerCompliance.label`.

Continuous live video is the API MJPEG route, not a Socket.IO dashboard.

## curl examples

```bash
curl http://127.0.0.1:8000/health

curl -F "file=@frame.jpg" "http://127.0.0.1:8000/predict/image?return_image=true"

curl -F "file=@clip.mp4" "http://127.0.0.1:8000/predict/video?return_clip=true&max_frames=120"
```
