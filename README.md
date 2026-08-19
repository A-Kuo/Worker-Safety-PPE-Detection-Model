# Worker Safety PPE Detection

Personal protective equipment detection for site cameras. A YOLOv8 detector on a
unified 14-class schema, an association step that ties each piece of gear to the
person wearing it, and an edge runtime that turns that into alerts you can act on.
The same runtime backs a CLI, a FastAPI service, and a Streamlit review UI.

| | |
|---|---|
| Training schema | Combined PPE v4, remapped to 14 unified classes (`helmet`, `no_helmet`, and so on) |
| Inherited baseline | Construction YOLOv8n, mAP50 0.809, mAP50-95 0.507 ([docs/baseline.md](docs/baseline.md)) |
| Runtime | [docs/edge_runtime.md](docs/edge_runtime.md) |
| Service and UI | [app/README.md](app/README.md) |
| Credits | [ATTRIBUTION.md](ATTRIBUTION.md) |

---

## 1. What this does

A detector that reports helmets and people as separate boxes has not told you
anything useful yet. The question on a site is who is missing gear, and whether
they have been missing it long enough to be worth an alert.

The pipeline answers that in five steps:

1. Run a detector over the frame (PyTorch on a workstation, ONNX Runtime on a device).
2. Map whatever vocabulary the checkpoint uses onto one unified label set.
3. Track people across frames so worker 3 stays worker 3.
4. Attach each PPE box to a person and compare against the required gear.
5. Debounce the result, so a violation raises once it persists and then goes quiet.

```python
from ppe import EdgePipeline, RuntimeConfig

pipeline = EdgePipeline.from_config(RuntimeConfig(weights="models/best.onnx"))
for result in pipeline.process_stream(frames):
    for event in result.events:
        print(event.describe())  # worker 3 now no_helmet (4 frames, 0.5s)
```

---

## 2. Install

```bash
python -m pip install -e ".[edge]"        # numpy, opencv, onnxruntime: runs on a device
python -m pip install -e ".[torch]"       # ultralytics + torch: training and .pt files
python -m pip install -e ".[app]"         # FastAPI service and Streamlit UI
python -m pip install -e ".[edge,app,dev]"  # everything the test suite needs
```

The core library depends on numpy and pyyaml only. Torch is an optional extra
because a Jetson or an Intel NUC running the ONNX path does not need it.

---

## 3. Command line

```bash
ppe classes                                     # print the 14-class schema
ppe image site.jpg --weights models/best.onnx   # score a still or a directory
ppe video clip.mp4 --save annotated.mp4         # score a clip, write an overlay
ppe watch rtsp://cam.local/stream               # follow a live camera, print alerts
ppe bench --frames 200 --json                   # latency and throughput
ppe serve --port 8000                           # start the HTTP service
```

`python -m ppe` works the same way if you would rather not install the script.
Every subcommand accepts `--backend {auto,ultralytics,onnx,stub}`, `--conf`,
`--iou`, `--imgsz`, `--device`, `--required`, `--stride`, `--alert-frames`, and
`--alert-cooldown`. Settings also come from `PPE_*` environment variables, and
the command line wins over the environment.

The `stub` backend replays scripted detections instead of loading a model, which
is how the test suite and a first deployment dry-run avoid needing weights.

---

## 4. Data

| Dataset | Role | Notes |
|---|---|---|
| [Construction Site Safety v28](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/28) | Inherited baseline train/eval | 10 classes; split 2605 / 114 / 82. A val split of 114 images cannot carry strong per-class claims |
| [PPE Combined Model v4](https://universe.roboflow.com/roboflow-universe-projects/personal-protective-equipment-combined-model/dataset/4) | Unified training set | ~44k images, 14 classes, 70/20/10 |
| [Hard Hat Universe](https://universe.roboflow.com/universe-datasets/hard-hat-universe-0dy7t) | Held-out helmet-domain eval | ~7k images, never mixed into training |

The Roboflow Universe sets are CC BY 4.0. See [ATTRIBUTION.md](ATTRIBUTION.md).

Label normalization, Combined to unified:

| Combined raw | Unified |
|---|---|
| Hardhat / NO-Hardhat | `helmet` / `no_helmet` |
| Safety Vest / NO-Safety Vest | `vest` / `no_vest` |
| Goggles / NO-Goggles | `goggles` / `no_goggles` |
| Gloves / NO-Gloves | `gloves` / `no_gloves` |
| Mask / NO-Mask | `mask` / `no_mask` |
| Person / Safety Cone | `person` / `cone` |
| Ladder / Fall-Detected | `ladder` / `fall_detected` |

`machinery` and `vehicle` stay on the Construction baseline; they are not part of
the unified model. Boots are out of scope because Combined has no boot classes.

Construction is never merged into Combined. Construction already clones imagery
from Combined and other Universe sets, so merging without perceptual hashing
would leak training images into evaluation. Each is remapped separately, and
Construction is scored on shared classes only.

Configs: [`configs/data/`](configs/data/). Counts: [`docs/data_distribution.md`](docs/data_distribution.md).

---

## 5. Model

- Detector: Ultralytics YOLOv8n. The inherited Construction weights live under
  [`baselines/snehilsanyal_yolov8n_css/`](baselines/snehilsanyal_yolov8n_css/);
  unified runs use the same family.
- Grid variants: YOLOv8s in E1, with YOLOv8m only if E1 and E4 finish early.
- Compliance: [`src/ppe/compliance.py`](src/ppe/compliance.py) attaches PPE boxes
  to people by containment or IoU and emits `Worker 3: missing helmet and vest`.
- Tracking: [`src/ppe/tracking.py`](src/ppe/tracking.py), greedy IoU matching, no
  motion model. Enough for fixed cameras and nearly free on a low-power device.

External reference, not a checkpoint from here:
[Hexmon/vyra-yolo-ppe-detection](https://huggingface.co/Hexmon/vyra-yolo-ppe-detection).

---

## 6. Training and experiments

| | |
|---|---|
| Configs | [`configs/train/`](configs/train/): `e0_n`, `e1_s`, `e2_focal`, `e3_augs`, `e4_full44k` |
| Entry point | `python scripts/train.py --exp e0_n` |
| Protocol | [`docs/experiments.md`](docs/experiments.md) |

One factor changes per run, on a fixed seed and split:

| ID | Change | Data |
|---|---|---|
| E0 | YOLOv8n defaults | Stratified Combined 12k subset |
| E1 | YOLOv8s | Same subset |
| E2 | `fl_gamma=1.5` (focal loss) | Same subset |
| E3 | Heavier augmentation (blur, brightness, crop) | Same subset |
| E4 | YOLOv8n, 50 epochs on the full 44k | Confirmation run, shipped detector |

Combined metrics are pending: they need `ROBOFLOW_API_KEY` for the downloads and
a GPU for the runs. When they land, 14-class Combined mAP and 10-class
Construction mAP will still measure different tasks; compare on
`SHARED_EVAL_CLASSES` instead.

---

## 7. Evaluation

The inherited Construction baseline, from its own `results.csv` at epoch 99:

| Metric | Value |
|---|---|
| mAP@0.50 | 0.809 |
| mAP@0.50:0.95 | 0.507 |
| Precision | 0.900 |
| Recall | 0.731 |

Its confusion matrix shows `NO-*` classes leaking into background, which is a
missed violation and the reason the calibration step sweeps confidence on those
classes specifically. Per-class numbers need Construction images on disk and a
run of `python scripts/eval_baseline.py`.

| Script | Purpose |
|---|---|
| `scripts/eval.py` | In-domain Combined eval |
| `scripts/eval_cross_domain.py` | Combined, Construction (mapped), and HHU tables |
| `scripts/calibrate.py` | ECE, Brier, and `no_*` confidence sweeps at recall 0.90 |
| `scripts/benchmark.py`, `scripts/export_onnx.py` | Latency, FPS, memory, PyTorch against ONNX |

---

## 8. Service and UI

```bash
python -m pip install -e ".[app]"
ppe serve --port 8000                       # or: uvicorn app.api.main:app --reload
streamlit run app/ui/streamlit_app.py       # second terminal
```

OpenAPI at http://127.0.0.1:8000/docs. Endpoints: `GET /health`, `GET /metrics`,
`GET /classes`, `POST /predict/image`, `POST /predict/video`, `GET /clips/{name}`,
`GET /stream` (MJPEG). Details and curl examples in [app/README.md](app/README.md).

Weights resolve from `PPE_WEIGHTS`, then
`baselines/snehilsanyal_yolov8n_css/models/best.pt`, then `models/best.pt`.

---

## 9. Tests

```bash
python -m pip install -e ".[edge,app,dev]"
pytest
ruff check . && ruff format --check .
```

The suite runs without a checkpoint, a GPU, or a camera. It builds a synthetic
ONNX graph to exercise the real ONNX Runtime path, generates video with OpenCV
for the source and API tests, and drives everything else through the stub
backend. CI runs the same three commands on Python 3.10 through 3.12.

---

## 10. Future work

- Appearance features in the tracker for crossing workers and long occlusions
- SCADA and MES hooks so alerts reach plant alarms and shift dashboards
- Infrared and thermal cameras for night shifts and low light
- INT8 quantization once E4 gives real latency numbers to compare against
- Frame retrieval over a labeled bank ("frames with missing vests") using a
  frozen image encoder, once the detector path is solid

---

## 11. Inherited versus built here

| Inherited | Built here |
|---|---|
| Snehil Sanyal's Construction YOLOv8n weights, plots, `results.csv`, sample media | `src/ppe/`: schema, compliance, tracking, events, backends, pipeline, CLI |
| The original Roboflow Construction notes and yaml layout | `scripts/`: download, remap, subset, analyze, train, eval, calibrate, export, benchmark |
| That artifact dump, relocated under `baselines/snehilsanyal_yolov8n_css/` | `configs/`, `docs/`, and the test suite |
| | The FastAPI service and Streamlit UI under `app/` |

`baselines/.../models/best.pt` was not trained here.

### Citations

- Snehil Sanyal, [Construction-Site-Safety-PPE-Detection](https://github.com/snehilsanyal/Construction-Site-Safety-PPE-Detection)
- The Roboflow Universe datasets above, CC BY 4.0
- Hexmon/vyra-yolo-ppe-detection, an external Combined v4 reference

Full license table: [ATTRIBUTION.md](ATTRIBUTION.md).

---

## Repo map

```text
src/ppe/           schema, compliance, tracking, events, backends, pipeline, cli
scripts/           dataset and training CLIs (see run_pipeline.md)
configs/data/      construction, combined, hardhat_eval
configs/train/     E0-E4 experiment yamls
app/               FastAPI service and Streamlit UI
docs/              baseline, experiments, data_distribution, edge_runtime
baselines/         inherited Construction artifacts
tests/             the pytest suite
```

### End-to-end dataset pipeline

Also in [`scripts/run_pipeline.md`](scripts/run_pipeline.md). Downloads need
`ROBOFLOW_API_KEY`; training and eval want a GPU.

```bash
python scripts/download_datasets.py --execute
python scripts/remap_labels.py --source data/raw/combined --out data/processed/combined --mapping combined
python scripts/remap_labels.py --source data/raw/hardhat --out data/processed/hardhat --mapping hhu
python scripts/remap_labels.py --source data/raw/construction --out data/processed/construction --mapping construction
python scripts/make_subset.py --source data/raw/combined --out data/raw/combined_12k --n 12000 --seed 42
python scripts/analyze_distribution.py
python scripts/train.py --exp e0_n
python scripts/eval.py --weights runs/train/e0_n/weights/best.pt
python scripts/calibrate.py --weights runs/train/e0_n/weights/best.pt
python scripts/export_onnx.py --weights runs/train/e0_n/weights/best.pt
python scripts/benchmark.py --weights runs/train/e0_n/weights/best.pt
```
