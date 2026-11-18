//! Time `decode` on a real-sized tensor: `cargo run --release --example bench_decode`.
//!
//! Uses (6, 8400) tensors, the shape of the shipped vest model's output, filled by a small
//! deterministic generator so the run needs no files: a *sparse* one (a few dozen candidate boxes,
//! like a real frame) and a *dense* stress case (thousands of candidates). Compare with
//! `python scripts/bench_native_decode.py`, which times the Python decode on the same tensor.

use ppe_core::{decode, LetterboxMeta};
use std::time::Instant;

/// The same linear congruential generator as scripts/bench_native_decode.py.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

fn tensor(channels: usize, anchors: usize, sparse: bool) -> Vec<f32> {
    let mut rng = Lcg(42);
    let mut t = vec![0.0f32; channels * anchors];
    for a in 0..anchors {
        t[a] = rng.next() * 640.0;
        t[anchors + a] = rng.next() * 640.0;
        t[2 * anchors + a] = 20.0 + rng.next() * 200.0;
        t[3 * anchors + a] = 20.0 + rng.next() * 200.0;
        for c in 4..channels {
            let mut u = rng.next();
            if sparse {
                // u^512 by repeated squaring: a few dozen candidates, like a real frame.
                for _ in 0..9 {
                    u *= u;
                }
            } else {
                u = u * u * u * u * u * u; // stress: ~3,000 candidates, quadratic NMS dominates
            }
            t[c * anchors + a] = u;
        }
    }
    t
}

fn main() {
    let (channels, anchors, iters) = (6usize, 8400usize, 2000usize);
    let meta = LetterboxMeta::for_image(1280, 720, 640);
    for (label, sparse) in [("sparse", true), ("dense ", false)] {
        let t = tensor(channels, anchors, sparse);
        let mut kept = 0usize;
        for _ in 0..50 {
            kept = decode(&t, channels, anchors, &meta, 0.25, 0.45).unwrap().len();
        }
        let mut times: Vec<f64> = (0..iters)
            .map(|_| {
                let start = Instant::now();
                let d = decode(std::hint::black_box(&t), channels, anchors, &meta, 0.25, 0.45).unwrap();
                std::hint::black_box(d);
                start.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        times.sort_by(|a, b| a.total_cmp(b));
        let checksum: f64 = t.iter().map(|&v| v as f64).sum();
        println!(
            "rust   {label} 6x8400 conf 0.25: detections={kept:4} p50={:.4} ms p95={:.4} ms tensor_sum={:.3}",
            times[iters / 2],
            times[iters * 95 / 100],
            checksum
        );
    }
}
