# native/ (experiment, incomplete)

A first look at moving the detector's hot path out of Python. **This is an experiment. It is not used by
the `ppe` package, the CLI, the window or the released wheel**, and nothing depends on it.

## What exists

`ppe-core`: a dependency-free Rust crate (Rust 1.77+) with a port of the post-processing step only:

| Piece | Python source | Status |
|---|---|---|
| Letterbox geometry (scale, padding) | `Letterboxer` in `src/ppe/runtime/backends.py` | Ported, exact match |
| Decode raw YOLOv8 output to boxes | `_decode_yolo_onnx` | Ported |
| Class-agnostic NMS | `_nms` | Ported |

```bash
cd native/ppe-core
cargo test                                   # unit tests + parity with the Python runtime
cargo run --release --example bench_decode
```

## How parity is checked

`scripts/gen_native_fixtures.py` builds raw detection tensors and records what the **Python** decode
returns for each (five cases: 2-class landscape and portrait, 14-class, nothing above threshold, a
small square input). `cargo test` decodes the same tensors and must return the same classes, with
confidences within 1e-6 and coordinates within 1e-3 px. `tests/test_native_fixtures.py` fails if the
committed fixtures no longer match the Python decode, so the two cannot drift apart silently.

## What the first measurement says

Same (6, 8400) tensor in both programs (identical checksum and detection count), Intel Core Ultra 7
256V, two runs each, p50:

| Tensor | Detections | Python (numpy) | Rust |
|---|---|---|---|
| Sparse: a few dozen candidate boxes, like a real frame | 42 | 0.46 ms | 0.010 ms |
| Dense: thousands of candidates (stress) | 1,471 | 24 ms | 3.5 ms |

Read this carefully before drawing conclusions:

- On a realistic frame the port saves about **0.45 ms**. A full frame on the NPU takes 4 to 6 ms, of
  which inference is 2.6 to 4.2 ms, so decode was never the bottleneck. The ratio is large; the
  absolute gain is small.
- The dense case matters only if a model emits thousands of candidates (a very low confidence
  threshold, or a broken model). There the quadratic NMS dominates in both languages.
- Single machine, not isolated; a micro-benchmark of one function, not an end-to-end result.

So this step alone does not justify a rewrite. The pieces that could are below.

## Not done

- **Preprocessing** (resize, pad, BGR to RGB, normalise): about 1.1 to 1.3 ms per frame today, the
  largest non-inference cost. Bit-exact parity with OpenCV's bilinear resize is the hard part.
- **ONNX Runtime session** and execution-provider selection (the `ort` crate or the C API), including
  strict-NPU mode and placement verification. Without this there is no native inference at all.
- Camera capture, the live window, model download and verification.
- A Python binding (PyO3 / cffi) so the existing package could call the native code.
- Windows packaging of a native library alongside the wheel.
- The C++ alternative has not been tried. No C++ compiler or CMake is installed on the development
  machine; Rust was.

## If this continues

1. Port preprocessing and measure the whole non-inference path end to end, not per function.
2. Decide Rust or C++ on ONNX Runtime integration, which is where the real work is.
3. Only then consider a binding into the Python package.
