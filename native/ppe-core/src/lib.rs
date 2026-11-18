//! Experimental native port of the detector's post-processing. **Incomplete by design.**
//!
//! Ported so far (and checked against the Python runtime through shared fixtures):
//! - letterbox geometry ([`LetterboxMeta::for_image`]), mirroring `Letterboxer` in
//!   `src/ppe/runtime/backends.py`;
//! - decoding a raw YOLOv8 output tensor to boxes ([`decode`]), mirroring `_decode_yolo_onnx`;
//! - class-agnostic greedy NMS ([`nms`]), mirroring `_nms`.
//!
//! Not ported: image resize / normalisation, the ONNX Runtime session and execution-provider
//! handling, strict-NPU verification, capture, the window. See `native/README.md`.
//!
//! Arithmetic is done in `f32` in the same order as the numpy code so results agree to float
//! rounding; the fixture tests compare with a small tolerance.

/// How a frame was fitted into the square model input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LetterboxMeta {
    /// Resize factor applied to the original image.
    pub scale: f64,
    /// Padding (pixels, in model-input space) added on the left and top.
    pub pad_left: i32,
    pub pad_top: i32,
    /// Original image size in pixels.
    pub orig_w: i32,
    pub orig_h: i32,
}

impl LetterboxMeta {
    /// Geometry for an `orig_w` x `orig_h` image letterboxed into `imgsz` x `imgsz`.
    ///
    /// Python computes the resized size with `int(round(x))`, which rounds halves to even;
    /// `round_ties_even` matches that (plain `round` would not).
    pub fn for_image(orig_w: i32, orig_h: i32, imgsz: i32) -> Self {
        let s = imgsz as f64;
        let scale = (s / orig_h as f64).min(s / orig_w as f64);
        let new_w = (orig_w as f64 * scale).round_ties_even() as i32;
        let new_h = (orig_h as f64 * scale).round_ties_even() as i32;
        LetterboxMeta {
            scale,
            pad_left: (imgsz - new_w) / 2,
            pad_top: (imgsz - new_h) / 2,
            orig_w,
            orig_h,
        }
    }
}

/// One detection in original-image pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    pub class_id: usize,
    pub conf: f32,
    /// `[x1, y1, x2, y2]`
    pub xyxy: [f32; 4],
}

/// Errors from [`decode`]. The Python code returns an empty list for a malformed tensor; here a
/// wrong length is a caller bug and is reported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// `output.len() != channels * anchors`.
    BadLength { expected: usize, got: usize },
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::BadLength { expected, got } => {
                write!(f, "output tensor has {got} values, expected {expected}")
            }
        }
    }
}

impl std::error::Error for DecodeError {}

/// Decode a raw YOLOv8 detection tensor laid out channels-first, `(4 + nc, anchors)`, row-major:
/// rows 0..4 are `cx, cy, w, h` in model-input pixels, rows 4.. are per-class scores.
///
/// Keeps anchors whose best class score is `>= conf`, maps boxes back to the original image and
/// applies class-agnostic NMS at `iou`. Results are in descending confidence.
pub fn decode(
    output: &[f32],
    channels: usize,
    anchors: usize,
    meta: &LetterboxMeta,
    conf: f32,
    iou: f32,
) -> Result<Vec<Detection>, DecodeError> {
    let expected = channels * anchors;
    if output.len() != expected {
        return Err(DecodeError::BadLength { expected, got: output.len() });
    }
    if channels < 5 {
        return Ok(Vec::new());
    }
    let nc = channels - 4;
    let at = |c: usize, a: usize| output[c * anchors + a];

    let scale = meta.scale as f32;
    let (left, top) = (meta.pad_left as f32, meta.pad_top as f32);
    let (w0, h0) = (meta.orig_w as f32, meta.orig_h as f32);

    let mut boxes: Vec<[f32; 4]> = Vec::new();
    let mut confs: Vec<f32> = Vec::new();
    let mut classes: Vec<usize> = Vec::new();
    for a in 0..anchors {
        // argmax over classes; the first maximum wins, like numpy.argmax.
        let mut best = 0usize;
        let mut best_score = at(4, a);
        for c in 1..nc {
            let s = at(4 + c, a);
            if s > best_score {
                best_score = s;
                best = c;
            }
        }
        // `!(x >= conf)` also drops NaN, as the numpy comparison does.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(best_score >= conf) {
            continue;
        }
        let (cx, cy, w, h) = (at(0, a), at(1, a), at(2, a), at(3, a));
        let x1 = ((cx - w / 2.0) - left) / scale;
        let y1 = ((cy - h / 2.0) - top) / scale;
        let x2 = ((cx + w / 2.0) - left) / scale;
        let y2 = ((cy + h / 2.0) - top) / scale;
        boxes.push([clip(x1, w0), clip(y1, h0), clip(x2, w0), clip(y2, h0)]);
        confs.push(best_score);
        classes.push(best);
    }

    Ok(nms(&boxes, &confs, iou)
        .into_iter()
        .map(|i| Detection { class_id: classes[i], conf: confs[i], xyxy: boxes[i] })
        .collect())
}

fn clip(v: f32, hi: f32) -> f32 {
    v.max(0.0).min(hi)
}

/// Greedy class-agnostic NMS. Returns indices into `boxes`, highest score first.
///
/// A box is suppressed when its IoU with an already kept box is `> iou_thresh`.
pub fn nms(boxes: &[[f32; 4]], scores: &[f32], iou_thresh: f32) -> Vec<usize> {
    assert_eq!(boxes.len(), scores.len(), "boxes and scores must have the same length");
    let areas: Vec<f32> = boxes
        .iter()
        .map(|b| (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0))
        .collect();
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    // Descending by score. Ties keep the later index first, like numpy's `argsort()[::-1]` does for
    // a stable sort; the fixtures contain no ties, so this choice is not exercised by them.
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(b.cmp(&a)));

    let mut keep = Vec::new();
    let mut suppressed = vec![false; boxes.len()];
    for (pos, &i) in order.iter().enumerate() {
        if suppressed[i] {
            continue;
        }
        keep.push(i);
        for &j in &order[pos + 1..] {
            if suppressed[j] {
                continue;
            }
            let xx1 = boxes[i][0].max(boxes[j][0]);
            let yy1 = boxes[i][1].max(boxes[j][1]);
            let xx2 = boxes[i][2].min(boxes[j][2]);
            let yy2 = boxes[i][3].min(boxes[j][3]);
            let inter = (xx2 - xx1).max(0.0) * (yy2 - yy1).max(0.0);
            let iou = inter / (areas[i] + areas[j] - inter + 1e-6);
            if iou > iou_thresh {
                suppressed[j] = true;
            }
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_geometry_matches_the_python_example() {
        // 128x96 into 64: scale 0.5, resized 64x48, padded 8 px top and bottom.
        let m = LetterboxMeta::for_image(128, 96, 64);
        assert_eq!(m.scale, 0.5);
        assert_eq!((m.pad_left, m.pad_top), (0, 8));
    }

    #[test]
    fn letterbox_rounds_half_to_even_like_python() {
        // 3x5 into 2: scale 0.4, width 1.2 -> 1, height 2.0. And 5x1 into 2: height 0.4 -> 0.
        assert_eq!(LetterboxMeta::for_image(5, 1, 2).pad_top, 1);
        // 25 * 0.1 = 2.5 rounds to 2 (even), not 3.
        let m = LetterboxMeta::for_image(100, 25, 10);
        assert_eq!(m.pad_top, (10 - 2) / 2);
    }

    #[test]
    fn decodes_and_unmaps_one_box() {
        // One anchor, two classes: box (20,20,16,16) in 64-space on a 128x96 image.
        let out = [20.0, 20.0, 16.0, 16.0, 0.9, 0.1];
        let m = LetterboxMeta::for_image(128, 96, 64);
        let d = decode(&out, 6, 1, &m, 0.25, 0.45).unwrap();
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].class_id, 0);
        assert_eq!(d[0].xyxy, [24.0, 8.0, 56.0, 40.0]);
    }

    #[test]
    fn threshold_is_inclusive_and_nan_is_dropped() {
        let m = LetterboxMeta::for_image(64, 64, 64);
        let at = [10.0, 10.0, 4.0, 4.0, 0.25];
        assert_eq!(decode(&at, 5, 1, &m, 0.25, 0.45).unwrap().len(), 1);
        let nan = [10.0, 10.0, 4.0, 4.0, f32::NAN];
        assert!(decode(&nan, 5, 1, &m, 0.25, 0.45).unwrap().is_empty());
    }

    #[test]
    fn nms_keeps_the_higher_score_and_non_overlapping_boxes() {
        let boxes = [[0.0, 0.0, 10.0, 10.0], [1.0, 0.0, 11.0, 10.0], [50.0, 50.0, 60.0, 60.0]];
        assert_eq!(nms(&boxes, &[0.8, 0.9, 0.5], 0.45), vec![1, 2]);
        assert!(nms(&[], &[], 0.45).is_empty());
    }

    #[test]
    fn boxes_are_clipped_to_the_image() {
        let m = LetterboxMeta::for_image(64, 64, 64);
        let out = [60.0, 2.0, 20.0, 20.0, 0.9];
        let d = decode(&out, 5, 1, &m, 0.25, 0.45).unwrap();
        assert_eq!(d[0].xyxy, [50.0, 0.0, 64.0, 12.0]);
    }

    #[test]
    fn wrong_length_is_an_error_and_too_few_channels_is_empty() {
        let m = LetterboxMeta::for_image(64, 64, 64);
        assert_eq!(
            decode(&[0.0; 5], 6, 1, &m, 0.25, 0.45),
            Err(DecodeError::BadLength { expected: 6, got: 5 })
        );
        assert!(decode(&[0.0; 4], 4, 1, &m, 0.25, 0.45).unwrap().is_empty());
    }
}
