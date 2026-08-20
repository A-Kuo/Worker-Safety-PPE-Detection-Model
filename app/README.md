# Service and UI

FastAPI for inference over HTTP, Streamlit for reviewing results by hand. Both
sit on top of `ppe.pipeline.EdgePipeline`; neither reimplements detection or
person-to-PPE association.

The service inherits the NPU-only execution policy. Starting it on a host with
no NPU execution provider fails on the first request rather than serving CPU
inference quietly. `GET /health` and `GET /devices` say which providers are
present; see [docs/npu_runtime.md](../docs/npu_runtime.md).

The process holds one pipeline and rebuilds it only when the weights path
changes, since loading a checkpoint costs seconds and a few hundred megabytes.
Inference is not reentrant, so `app.runtime.PIPELINE_LOCK` guards it.

## Install

```bash
python -m pip install -e ".[edge,app]"
python -m pip install onnxruntime-qnn   # or -openvino, -vitisai, -directml
ppe devices
```

That covers `fastapi`, `uvicorn`, `python-multipart`, `streamlit`, and
`opencv-python-headless`. There is no torch install for the service: it serves
ONNX models and nothing else.

## Weights

`PPE_WEIGHTS` wins. Failing that, the first of these that exists:

1. `models/best.int8.onnx`
2. `models/best.onnx`
3. `baselines/snehilsanyal_yolov8n_css/models/best.pt`

The `.pt` resolves last only so `/health` can report that it is the wrong
format. Export and quantize it first; see
[docs/npu_runtime.md](../docs/npu_runtime.md).

Other variables: `PPE_EXECUTION` (default `npu`), `PPE_PROVIDER`,
`PPE_PROVIDER_OPTIONS`, `PPE_API_HOST` (default `127.0.0.1`), `PPE_API_PORT`
(default `8000`), and everything else in
[docs/edge_runtime.md](../docs/edge_runtime.md).

```powershell
$env:PPE_WEIGHTS = "models/best.int8.onnx"
$env:PPE_PROVIDER = "QNNExecutionProvider"
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
| `GET` | `/health` | Ready flag, weights path, backend, execution policy, bound provider |
| `GET` | `/devices` | Every execution provider in the registry and whether this host has it |
| `GET` | `/metrics` | Rolling latency, bound provider, open tracks, active violations |
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
curl http://127.0.0.1:8000/devices

curl -F "file=@frame.jpg" "http://127.0.0.1:8000/predict/image?return_image=true"

curl -F "file=@clip.mp4" "http://127.0.0.1:8000/predict/video?return_clip=true&max_frames=120"

curl http://127.0.0.1:8000/metrics
```

## Troubleshooting

| Symptom | Cause | Do this |
|---|---|---|
| Every predict call is 503 | No NPU provider, strict policy | `curl /devices`, install the wheel, or set `PPE_EXECUTION=npu-preferred` |
| `/health` says `not an ONNX model` | `PPE_WEIGHTS` points at a `.pt` | Export and quantize first |
| `/metrics` shows `CPUExecutionProvider` | The accelerator did not bind | See the npu_runtime troubleshooting table |
| `/stream` returns 503 | Camera or URL could not be opened | Check the source; headless hosts have no webcam |
