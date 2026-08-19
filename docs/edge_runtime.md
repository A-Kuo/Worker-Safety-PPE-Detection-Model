# Edge runtime

How a frame becomes an alert, and which knob controls which stage.

## The path

```
frame -> backend -> unify labels -> tracker -> compliance -> alert monitor
```

| Stage | Module | What it produces |
|---|---|---|
| Inference | `ppe.backends` | `RawDetection` boxes in the model's own vocabulary |
| Label mapping | `ppe.schema.unify_name` | The unified 14-class vocabulary |
| Tracking | `ppe.tracking.IouTracker` | A stable id per person across frames |
| Association | `ppe.compliance` | `WorkerCompliance` per person: present, missing, violations |
| Alerting | `ppe.events.ViolationMonitor` | `ViolationEvent`, debounced and rate-limited |

`ppe.pipeline.EdgePipeline` owns all five and records latency as it goes.

## Backends

| Name | Needs | Use it for |
|---|---|---|
| `ultralytics` | `ultralytics`, `torch` | `.pt` checkpoints, training, a workstation |
| `onnx` | `onnxruntime` | Devices. No torch, and the postprocessing is plain numpy |
| `stub` | nothing | Tests, and wiring a deployment before weights exist |

`backend="auto"` reads the weights extension: `.onnx` picks the ONNX path,
anything else picks ultralytics.

The ONNX path does its own work rather than borrowing Ultralytics' helpers, all
of it in `ppe.postprocess`:

1. `letterbox` resizes to a square canvas, preserving aspect ratio, padding grey.
2. `to_input_tensor` produces a normalized NCHW RGB batch.
3. `decode_yolo_output` reads the `(1, 4 + num_classes, num_anchors)` head, keeps
   anchors above the confidence threshold, and runs class-aware NMS.
4. `undo_letterbox` maps boxes back onto the source frame and clips them to it.

Pass `num_classes` to the decoder when you know it. Without that hint the layout
is guessed from which axis is longer, which is right for a real export with
8400 anchors and wrong for a two-anchor test tensor.

## Configuration

`RuntimeConfig` is frozen and validated on construction, so a bad threshold
fails at startup rather than a thousand frames in. Every field has a `PPE_*`
environment variable, and `config_from_env()` reads them.

| Field | Env | Default | Effect |
|---|---|---|---|
| `weights` | `PPE_WEIGHTS` | none | Checkpoint path |
| `backend` | `PPE_BACKEND` | `auto` | Which implementation to load |
| `device` | `PPE_DEVICE` | auto | `cpu`, `cuda`, or an index (ultralytics only) |
| `imgsz` | `PPE_IMGSZ` | 640 | Inference size, a multiple of 32 |
| `conf` | `PPE_CONF` | 0.25 | Confidence floor |
| `iou` | `PPE_IOU` | 0.45 | NMS overlap threshold |
| `required_ppe` | `PPE_REQUIRED_PPE` | `helmet,vest` | Gear every worker must wear |
| `track_iou` | `PPE_TRACK_IOU` | 0.3 | Overlap needed to continue a track |
| `track_max_age` | `PPE_TRACK_MAX_AGE` | 15 | Frames a track survives without a match |
| `alert_min_frames` | `PPE_ALERT_FRAMES` | 3 | Consecutive frames before an alert |
| `alert_cooldown_s` | `PPE_ALERT_COOLDOWN` | 10.0 | Seconds of quiet after one |
| `frame_stride` | `PPE_FRAME_STRIDE` | 1 | Process every Nth frame |

## Tuning for a camera

**Site with intermittent occlusion.** Raise `track_max_age` so a worker walking
behind a pillar keeps their id, and raise `alert_min_frames` so the momentary
loss of a helmet box does not raise.

**Crowded frame.** Lower `track_iou`; people standing close produce boxes that
overlap each other more than they overlap their own previous position.

**Device that cannot keep up.** Raise `frame_stride` before you lower `imgsz`.
Skipping frames costs temporal resolution, which the debounce window already
smooths over; shrinking the input costs small-object recall, and a helmet at
thirty metres is a small object. Check `pipeline.stats()["latency"]` to see
which one you actually need.

**Noisy alerts.** Raise `alert_cooldown_s` first. If the same worker flickers in
and out of compliance, that is usually a detection problem rather than an
alerting one; check `conf` against the calibration sweeps in `docs/experiments.md`.

## Latency

`EdgePipeline` keeps a 200-frame rolling window:

```python
pipeline.stats()
# {'backend': 'onnx', 'frames': 412, 'tracks_open': 3,
#  'active_violations': 1,
#  'latency': {'frames': 412, 'window': 200, 'mean_ms': 24.1,
#              'p50_ms': 23.4, 'p95_ms': 31.8, 'max_ms': 58.2, 'fps': 41.5}}
```

p95 matters more than the mean here. A stream that drops a frame every twenty is
a stream that misses violations, and the mean hides that.

`ppe bench` reports the same numbers over synthetic frames, or over a real clip
with `--source`. Call `pipeline.warmup(frame)` before timing anything: the first
inference through a fresh session is not representative.

## Stateful and stateless calls

`EdgePipeline.process` is stateful by design, since tracking and debouncing are
what make it useful. Anything scoring an unrelated image has to call `reset()`
first, or the still inherits track ids and alert streaks from whatever came
before it. `PPEDetector`, `POST /predict/image`, and `ppe image` all do this.

`GET /stream` deliberately does not: a live stream is exactly the case the
tracker is for.
