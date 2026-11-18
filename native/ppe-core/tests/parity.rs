//! Parity with the Python runtime: decode the fixtures written by `scripts/gen_native_fixtures.py`
//! and compare with the detections Python produced for the same tensor.

use ppe_core::{decode, LetterboxMeta};
use std::fs;
use std::path::PathBuf;

struct Fixture {
    name: String,
    channels: usize,
    anchors: usize,
    imgsz: i32,
    meta: LetterboxMeta,
    conf: f32,
    iou: f32,
    tensor: Vec<f32>,
    expect: Vec<(usize, f32, [f32; 4])>,
}

fn parse(text: &str) -> Fixture {
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    let mut field = |key: &str| -> Vec<String> {
        let line = lines.next().unwrap_or_else(|| panic!("missing `{key}`"));
        let mut parts = line.split_whitespace();
        assert_eq!(parts.next(), Some(key), "expected `{key}` in `{line}`");
        parts.map(str::to_string).collect()
    };
    let name = field("name")[0].clone();
    let channels: usize = field("channels")[0].parse().unwrap();
    let anchors: usize = field("anchors")[0].parse().unwrap();
    let imgsz: i32 = field("imgsz")[0].parse().unwrap();
    let orig: Vec<i32> = field("orig").iter().map(|v| v.parse().unwrap()).collect();
    let scale: f64 = field("scale")[0].parse().unwrap();
    let pad: Vec<i32> = field("pad").iter().map(|v| v.parse().unwrap()).collect();
    let conf: f32 = field("conf")[0].parse().unwrap();
    let iou: f32 = field("iou")[0].parse().unwrap();
    assert!(field("tensor").is_empty());

    let mut tensor = Vec::with_capacity(channels * anchors);
    for _ in 0..channels {
        let row = lines.next().expect("tensor row");
        tensor.extend(row.split_whitespace().map(|v| v.parse::<f32>().unwrap()));
    }
    let header = lines.next().expect("expect line");
    let n: usize = header.strip_prefix("expect ").expect("expect N").parse().unwrap();
    let expect = (0..n)
        .map(|_| {
            let v: Vec<&str> = lines.next().expect("detection").split_whitespace().collect();
            let f = |i: usize| v[i].parse::<f32>().unwrap();
            (v[0].parse().unwrap(), f(1), [f(2), f(3), f(4), f(5)])
        })
        .collect();
    assert!(lines.next().is_none(), "trailing lines in fixture {name}");

    let meta = LetterboxMeta { scale, pad_left: pad[0], pad_top: pad[1], orig_w: orig[0], orig_h: orig[1] };
    Fixture { name, channels, anchors, imgsz, meta, conf, iou, tensor, expect }
}

fn fixtures() -> Vec<Fixture> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures");
    let mut paths: Vec<_> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    paths.sort();
    assert!(paths.len() >= 5, "expected the generated fixtures in {}", dir.display());
    paths.iter().map(|p| parse(&fs::read_to_string(p).unwrap().replace("\r\n", "\n"))).collect()
}

#[test]
fn decode_matches_the_python_runtime() {
    let mut total = 0;
    for fx in fixtures() {
        assert_eq!(fx.tensor.len(), fx.channels * fx.anchors, "{}", fx.name);
        let got = decode(&fx.tensor, fx.channels, fx.anchors, &fx.meta, fx.conf, fx.iou).unwrap();
        assert_eq!(got.len(), fx.expect.len(), "{}: number of detections", fx.name);
        for (i, (d, (cls, conf, xyxy))) in got.iter().zip(&fx.expect).enumerate() {
            assert_eq!(d.class_id, *cls, "{} #{i}: class", fx.name);
            assert!((d.conf - conf).abs() <= 1e-6, "{} #{i}: conf {} vs {}", fx.name, d.conf, conf);
            #[allow(clippy::needless_range_loop)]
            for k in 0..4 {
                // Coordinates reach ~1300 px; f32 has ~1e-4 resolution there.
                let tol = 1e-3_f32.max(xyxy[k].abs() * 2e-6);
                assert!(
                    (d.xyxy[k] - xyxy[k]).abs() <= tol,
                    "{} #{i}: coord {k} {} vs {}",
                    fx.name, d.xyxy[k], xyxy[k]
                );
            }
        }
        total += got.len();
    }
    assert!(total > 20, "fixtures should exercise real detections, got {total}");
}

#[test]
fn letterbox_geometry_matches_the_python_runtime() {
    for fx in fixtures() {
        let m = LetterboxMeta::for_image(fx.meta.orig_w, fx.meta.orig_h, fx.imgsz);
        assert_eq!(m, fx.meta, "{}", fx.name);
    }
}

#[test]
fn results_are_sorted_by_descending_confidence() {
    for fx in fixtures() {
        let got = decode(&fx.tensor, fx.channels, fx.anchors, &fx.meta, fx.conf, fx.iou).unwrap();
        assert!(got.windows(2).all(|w| w[0].conf >= w[1].conf), "{}", fx.name);
    }
}
