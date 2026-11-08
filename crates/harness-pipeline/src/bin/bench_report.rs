//! Native throughput report -> `benches/criterion_summary.json`.
//!
//! Measures RAW (non-criterion) microbenchmarks for the 3-gram spectrum
//! extraction, the Gaussian-RBF MMD² kernel, and ringbuffer deserialization,
//! then serializes a structured JSON baseline to `benches/criterion_summary.json`.

use std::time::Instant;

use harness_core::gate0::{mmd2, spectrum, Stream};
use rand::{rngs::SmallRng, Rng, SeedableRng};

const MOTIFS: &[&[u8]] = &[b"abba", b"aaba", b"baab"];

#[derive(serde::Serialize)]
struct Summary {
    spectrum_events_per_sec: f64,
    mmd2_per_sec: f64,
    mmd2_usec_per_call: f64,
    ringbuf_events_per_sec: f64,
    gate0_events_per_sec: f64,
    note: String,
}

fn make_prod(rng: &mut SmallRng, n: usize, len: usize) -> Vec<Stream> {
    (0..n)
        .map(|_| {
            let m = MOTIFS[rng.gen_range(0..MOTIFS.len())];
            (0..len).map(|i| m[i % m.len()]).collect()
        })
        .collect()
}

fn bench_spectrum() -> f64 {
    let mut rng = SmallRng::seed_from_u64(0);
    let streams = make_prod(&mut rng, 2000, 64);
    // warmup
    for s in streams.iter().take(50) {
        let _ = spectrum(s);
    }
    let t0 = Instant::now();
    let mut iters = 0u64;
    loop {
        for s in &streams {
            std::hint::black_box(spectrum(s));
        }
        iters += streams.len() as u64;
        if t0.elapsed().as_secs_f64() >= 3.0 {
            break;
        }
    }
    iters as f64 / t0.elapsed().as_secs_f64()
}

fn bench_mmd2() -> (f64, f64) {
    let mut rng = SmallRng::seed_from_u64(0);
    let a = make_prod(&mut rng, 48, 64);
    let b = make_prod(&mut rng, 48, 64);
    for _ in 0..5 {
        let _ = mmd2(&a, &b, 0.10);
    }
    let t0 = Instant::now();
    let mut iters = 0u64;
    loop {
        std::hint::black_box(mmd2(&a, &b, 0.10));
        iters += 1;
        if t0.elapsed().as_secs_f64() >= 4.0 {
            break;
        }
    }
    let per_sec = iters as f64 / t0.elapsed().as_secs_f64();
    (per_sec, 1e6 / per_sec)
}

#[derive(Copy, Clone)]
#[repr(C)]
struct NearMissEvent {
    pid: u32,
    tid: u32,
    kind: u32,
    tier: u32,
    token_digest: u64,
    ts_ns: u64,
}

fn bench_ringbuf() -> f64 {
    let ev = NearMissEvent { pid: 1, tid: 2, kind: 0, tier: 2, token_digest: 0xdead_beef, ts_ns: 123 };
    let size = std::mem::size_of::<NearMissEvent>();
    let n_events = 4096usize;
    let mut raw = Vec::with_capacity(n_events * size);
    for _ in 0..n_events {
        let slice: &[u8] = unsafe {
            std::slice::from_raw_parts(
                (&ev as *const NearMissEvent) as *const u8,
                size,
            )
        };
        raw.extend_from_slice(slice);
    }
    let reps = 200_000u64;
    let t0 = Instant::now();
    let mut acc = 0u64;
    for _ in 0..reps {
        for chunk in raw.chunks_exact(size) {
            acc += unsafe {
                std::ptr::read_unaligned(chunk.as_ptr() as *const NearMissEvent).token_digest
            };
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    std::hint::black_box(acc);
    (reps * (raw.len() / size) as u64) as f64 / secs
}

fn bench_gate0() -> f64 {
    let mut rng = SmallRng::seed_from_u64(1);
    let prod = make_prod(&mut rng, 90, 60);
    for _ in 0..5 {
        let _ = harness_core::gate0::gate0(&prod, &prod);
    }
    let t0 = Instant::now();
    let mut n = 0u64;
    while t0.elapsed().as_secs_f64() < 4.0 {
        let _ = harness_core::gate0::gate0(&prod, &prod);
        n += 1;
    }
    n as f64 / t0.elapsed().as_secs_f64()
}

fn main() {
    let spec_per_sec = bench_spectrum();
    let (mmd2_per_sec, mmd2_usec) = bench_mmd2();
    let ringbuf_per_sec = bench_ringbuf();
    let gate0_per_sec = bench_gate0();

    let summary = Summary {
        spectrum_events_per_sec: spec_per_sec,
        mmd2_per_sec,
        mmd2_usec_per_call: mmd2_usec,
        ringbuf_events_per_sec: ringbuf_per_sec,
        gate0_events_per_sec: gate0_per_sec,
        note: "Rust native (release) microbenchmarks; compare against the Python NumPy baseline proportionally.".into(),
    };

    let json = serde_json::to_string_pretty(&summary).unwrap();
    std::fs::write("benches/criterion_summary.json", json).unwrap();

    println!("[Task7] wrote benches/criterion_summary.json");
    println!("[Task7] spectrum {:.0} events/s", spec_per_sec);
    println!("[Task7] mmd2 {:.1} calls/s ({:.2} us/call)", mmd2_per_sec, mmd2_usec);
    println!("[Task7] ringbuf {:.0} events/s", ringbuf_per_sec);
    println!("[Task7] gate0 {:.0} evaluations/s", gate0_per_sec);
}