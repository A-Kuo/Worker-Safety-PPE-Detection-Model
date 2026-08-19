# Service and UI

FastAPI for inference over HTTP, Streamlit for reviewing results by hand. Both
sit on top of `ppe.pipeline.EdgePipeline`; neither reimplements detection or
person-to-PPE association.

The process holds one pipeline and rebuilds it only when the weights path
changes, since loading a checkpoint costs seconds and a few hundred megabytes.
Inference is not reentrant, so `app.runtime.PIPELINE_LOCK` guards it.

## Install

```bash
python -m pip install -e ".[edge,app]"
```

That covers `fastapi`, `uvicorn`, `python-multipart`, `streamlit`, and
`opencv-python-headless`. Add `".[torch]"` on top if you are serving a `.pt`
checkpoint rather than an ONNX export.

## Weights

`PPE_WEIGHTS` wins. Failing that, the first of these that exists:

1. `baselines/snehilsanyal_yolov8n_css/models/best.pt`
2. `models/best.pt`

Other variables: `PPE_DEVICE` (`cpu`, `cuda`, an index), `PPE_API_HOST`
(default `127.0.0.1`), `PPE_API_PORT` (default `8000`), and everything else in
[docs/edge_runtime.md](../docs/edge_runtime.md).

```powershell
$env:PPE_WEIGHTS = "models/best.onnx"
```

## Run the API

```bash
ppe serve --port 8000
# or
uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8000
# or
python -m app.api.main
```

OpenAPI: http://127.0.0.1:8000/docs

### Endpoints

| Method | Path | What it does |
|---|---|---|
| `GET` | `/health` | Ready flag, resolved weights path, selected backend, device |
| `GET` | `/metrics` | Rolling latency, open tracks, active violations |
| `GET` | `/classes` | The 14-class schema |
| `POST` | `/predict/image` | Multipart image to JSON detections and compliance. `?return_image=true` adds a base64 JPEG |
| `POST` | `/predict/video` | Multipart video, strided and capped at 300 frames / 45s. `?return_clip=true` writes `output/app/*.mp4` and returns `annotated_url` |
| `GET` | `/clips/{name}` | Download a clip written by `/predict/video` |
| `GET` | `/stream` | MJPEG. `?source=0` for a local camera, an `rtsp://` URL, or a file path |

Image query parameters: `conf`, `return_image`.
Video: `conf`, `return_clip`, `max_frames`, `max_seconds`.
Stream: `source`, `conf`, `max_fps` (default 12).

The upload endpoints reset the pipeline per request, so a still never inherits
track ids from an earlier one. `/stream` keeps its state, which is what makes
alert debouncing work on a live feed.

CORS is open. This is a local demo, not a deployment posture.

### When a source cannot be opened

`GET /stream` opens the source before it starts streaming. A missing camera, a
bad RTSP URL, or a path that is not there returns 503 with the reason. Headless
hosts have no camera and will always 503 on a numeric index.

## Run the UI

```bash
streamlit run app/ui/streamlit_app.py
```

Sidebar sets the weights path, confidence, and the frame cap. Tabs cover Image,
Video, and Webcam (a single `st.camera_input` snapshot). Each run shows the
annotated frame, per-worker compliance, and any alerts the monitor raised.

Continuous live video is the API's MJPEG route, not a Streamlit widget.

## curl

```bash
curl http://127.0.0.1:8000/health

curl -F "file=@frame.jpg" "http://127.0.0.1:8000/predict/image?return_image=true"

curl -F "file=@clip.mp4" "http://127.0.0.1:8000/predict/video?return_clip=true&max_frames=120"

curl http://127.0.0.1:8000/metrics
```
