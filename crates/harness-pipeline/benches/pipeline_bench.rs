//! Criterion micro-benchmarks: SIMD-friendly 3-gram spectrum extraction,
//! Gaussian-RBF MMD² kernel computation, IPW evaluation, and zero-copy
//! ringbuffer deserialization throughput.

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use harness_core::evaluator::evaluate_ipw;
use harness_core::gate0::{gate0, mmd2, spectrum, Stream};
use rand::{rngs::SmallRng, Rng, SeedableRng};

const MOTIFS: &[&[u8]] = &[b"abba", b"aaba", b"baab"];

fn make_prod(rng: &mut SmallRng, n: usize, len: usize) -> Vec<Stream> {
    (0..n)
        .map(|_| {
            let m = MOTIFS[rng.gen_range(0..MOTIFS.len())];
            (0..len).map(|i| m[i % m.len()]).collect()
        })
        .collect()
}

/// NearMissEvent layout mirroring the eBPF program (32 bytes, zero-copy).
#[repr(C)]
#[derive(Copy, Clone)]
struct NearMissEvent {
    pid: u32,
    tid: u32,
    kind: u32,
    tier: u32,
    token_digest: u64,
    ts_ns: u64,
}

fn spectrum_throughput(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(0);
    let streams = make_prod(&mut rng, 1000, 64);
    c.bench_function("spectrum_3gram_extraction/s", |b| {
        b.iter(|| streams.iter().map(|s| spectrum(s)).collect::<Vec<_>>())
    });
}

fn mmd2_kernel(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(0);
    let prod = make_prod(&mut rng, 48, 64);
    let cand = make_prod(&mut rng, 48, 64);
    c.bench_function("rbf_mmd2_kernel_n48", |b| {
        b.iter_batched(
            || (prod.clone(), cand.clone()),
            |(p, c)| mmd2(&p, &c, 0.10),
            BatchSize::SmallInput,
        )
    });
}

fn gate0_bench(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(1);
    let prod = make_prod(&mut rng, 90, 60);
    let cand = make_prod(&mut rng, 90, 60);
    c.bench_function("gate0_full_n90", |b| {
        b.iter_batched(
            || (prod.clone(), cand.clone()),
            |(p, c)| gate0(&p, &c),
            BatchSize::SmallInput,
        )
    });
}

fn ringbuf_deser(c: &mut Criterion) {
    let ev = NearMissEvent { pid: 1, tid: 2, kind: 0, tier: 2, token_digest: 0xdead_beef, ts_ns: 123 };
    let bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts((&ev as *const NearMissEvent) as *const u8, std::mem::size_of::<NearMissEvent>())
    }.to_vec();
    let bytes = criterion::black_box(bytes);
    c.bench_function("ringbuf_deserialize_events/s", |b| {
        b.iter(|| {
            let buf = criterion::black_box(bytes.as_slice());
            buf.chunks_exact(std::mem::size_of::<NearMissEvent>())
                .map(|chunk| unsafe {
                    let ev = std::ptr::read_unaligned(chunk.as_ptr() as *const NearMissEvent);
                    ev.token_digest ^ ev.ts_ns ^ ev.pid as u64
                })
                .fold(0u64, |a, x| a.wrapping_add(x))
        })
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

criterion_group!(pipelines, spectrum_throughput, mmd2_kernel, gate0_bench, ringbuf_deser, evaluator_bench);
criterion_main!(pipelines);