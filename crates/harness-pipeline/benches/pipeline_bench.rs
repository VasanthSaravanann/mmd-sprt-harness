//! Criterion benchmarks for the pipeline kernels.
//!
//! These measure telemetry throughput (streams/sec through Gate 0), allocation
//! count/bytes (via the `alloc` profiler channel), and IPW evaluation latency.
//! The Python baseline is compared in prose in the benchmark plots/`baselines`
//! table; this file quantifies the Rust side.

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use harness_core::evaluator::evaluate_ipw;
use harness_core::gate0::{gate0, Stream};
use rand::{rngs::SmallRng, Rng, SeedableRng};

const MOTIFS: &[&[u8]] = &[b"abba", b"aaba", b"baab"];

fn make_prod(rng: &mut SmallRng, n: usize) -> Vec<Stream> {
    (0..n)
        .map(|_| {
            let m = MOTIFS[rng.gen_range(0..MOTIFS.len())];
            (0..60).map(|i| m[i % m.len()]).collect()
        })
        .collect()
}

fn gate0_bench(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(0);
    let prod = make_prod(&mut rng, 90);
    let cand = make_prod(&mut rng, 90);
    c.bench_function("gate0_mmd2_n90", |b| {
        b.iter_batched(
            || (prod.clone(), cand.clone()),
            |(p, c)| gate0(&p, &c),
            BatchSize::SmallInput,
        )
    });
}

fn evaluator_bench(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(1);
    let px: Vec<f64> = (0..3000).map(|_| rng.gen_range(0.0..1.0)).collect();
    let py: Vec<f64> = px.iter().map(|x| 0.9 * x).collect();
    let pp = vec![0.01; px.len()];
    let ex: Vec<f64> = (0..4000).map(|_| rng.gen_range(0.0..1.0)).collect();
    let ey: Vec<f64> = ex.iter().map(|x| 0.95 * x).collect();
    let ep = vec![0.4; ex.len()];
    c.bench_function("ipw_spearman_n7k", |b| {
        b.iter(|| evaluate_ipw(&px, &py, &pp, &ex, &ey, &ep))
    });
}

criterion_group!(pipelines, gate0_bench, evaluator_bench);
criterion_main!(pipelines);