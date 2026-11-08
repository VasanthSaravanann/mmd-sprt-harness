//! Gate 0: distributional attestation.
//!
//! Compares a *candidate* stream set against a locked *production* set using a
//! length-normalized 3-gram spectrum, a Gaussian-RBF
//! ![MMD²](https://latex.codecogs.com/svg.image?\epsilon_{prod}=\max(\text{MMD}^2_\text{prod-prod},0.015))
//! floor, and an Action Jaccard distance `D_act < 0.05`.
//!
//! The crux property the tests verify: a 1% *tail-exfiltration* perturbation
//! leaves the mean-embedding cosine distance tiny (a cosine-only detector
//! "passes" it) while the 3-gram MMD² rises above `eps_prod` and flips the gate
//! to `GATE0_FAIL`.

use std::collections::HashMap;

pub const NGRAM_N: usize = 3;
pub const MMD2_FLOOR: f64 = 0.015;
pub const D_ACT_THRESHOLD: f64 = 0.05;
/// Gaussian-RBF bandwidth over the normalized n-gram spectrum.
pub const SIGMA_BW: f64 = 0.15;

/// A stream is a sequence of tokens represented as bytes.
pub type Stream = Vec<u8>;
/// A length-normalized 3-gram spectrum: gram -> probability mass.
pub type Spectrum = HashMap<[u8; NGRAM_N], f64>;

/// Result of the Gate 0 attestation.
pub struct Gate0Result {
    pub mmd2: f64,
    pub eps_prod: f64,
    pub d_act: f64,
    pub cosine_dist: f64,
    pub passed: bool,
    pub reason: String,
}

/// Build the length-normalized 3-gram spectrum for one stream.
///
/// The output sums to 1.0 (a proper probability distribution over the stream's
/// 3-gram repertoire), which makes streams of different lengths comparable.
pub fn spectrum(stream: &[u8]) -> Spectrum {
    let mut counts: HashMap<[u8; NGRAM_N], u64> = HashMap::new();
    if stream.len() < NGRAM_N {
        // degenerate: emit the whole stream as a single gram so vectors stay
        // well-formed (preventing a divide-by-zero / empty-distribution issue).
        let mut g = [0u8; NGRAM_N];
        for (i, b) in stream.iter().enumerate() {
            g[i] = *b;
        }
        counts.insert(g, 1);
    } else {
        for w in stream.windows(NGRAM_N) {
            let mut g = [0u8; NGRAM_N];
            g.copy_from_slice(w);
            *counts.entry(g).or_insert(0) += 1;
        }
    }
    let total: u64 = counts.values().sum();
    let total = total.max(1);
    counts.into_iter().map(|(g, c)| (g, c as f64 / total as f64)).collect()
}

fn squared_l2(a: &Spectrum, b: &Spectrum) -> f64 {
    // iterate over the smaller map for cache friendliness
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let mut d2 = 0.0;
    for (k, va) in small {
        let vb = large.get(k).copied().unwrap_or(0.0);
        let d = va - vb;
        d2 += d * d;
    }
    for (k, vb) in large {
        if !small.contains_key(k) {
            d2 += vb * vb;
        }
    }
    d2
}

/// Gaussian-RBF kernel over two spectra.
pub fn rbf_kernel(a: &Spectrum, b: &Spectrum, sigma: f64) -> f64 {
    let d2 = squared_l2(a, b);
    (-d2 / (2.0 * sigma * sigma)).exp()
}

/// Biased/quadratic MMD² between two multisets of streams, over the RBF kernel.
pub fn mmd2(x_streams: &[Stream], y_streams: &[Stream], sigma: f64) -> f64 {
    let fx: Vec<Spectrum> = x_streams.iter().map(|s| spectrum(s)).collect();
    let fy: Vec<Spectrum> = y_streams.iter().map(|s| spectrum(s)).collect();
    let m = fx.len();
    let n = fy.len();
    if m == 0 && n == 0 {
        return 0.0;
    }
    if m == 0 || n == 0 {
        return f64::INFINITY;
    }
    let fm = m as f64;
    let fn_count = n as f64;

    // parallelize the pairwise sums with rayon
    use rayon::prelude::*;
    let xx: f64 = fx
        .par_iter()
        .map(|fi| {
            fx.iter().map(|fj| rbf_kernel(fi, fj, sigma)).sum::<f64>()
        })
        .sum();
    let yy: f64 = fy
        .par_iter()
        .map(|fi| fy.iter().map(|fj| rbf_kernel(fi, fj, sigma)).sum::<f64>())
        .sum();
    let xy: f64 = fx
        .par_iter()
        .map(|fi| fy.iter().map(|fj| rbf_kernel(fi, fj, sigma)).sum::<f64>())
        .sum();

    xx / (fm * fm) + yy / (fn_count * fn_count) - 2.0 * xy / (fm * fn_count)
}

/// Mean spectrum embedding over a multiset (used for the cosine check).
pub fn mean_embedding(streams: &[Stream]) -> Spectrum {
    let vecs: Vec<Spectrum> = streams.iter().map(|s| spectrum(s)).collect();
    let mut out: Spectrum = HashMap::new();
    if vecs.is_empty() {
        return out;
    }
    let m = vecs.len() as f64;
    for v in &vecs {
        for (k, val) in v {
            *out.entry(*k).or_insert(0.0) += *val / m;
        }
    }
    out
}

fn norm(v: &Spectrum) -> f64 {
    v.values().map(|x| x * x).sum::<f64>().sqrt()
}

/// Cosine distance (1 - cosine similarity) between two mean embeddings.
pub fn cosine_distance(a: &Spectrum, b: &Spectrum) -> f64 {
    let na = norm(a);
    let nb = norm(b);
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    let dot: f64 = a.iter().map(|(k, va)| va * b.get(k).copied().unwrap_or(0.0)).sum();
    1.0 - dot / (na * nb)
}

/// The set of distinct 3-grams appearing across a multiset ("actions").
fn action_set(streams: &[Stream]) -> std::collections::HashSet<[u8; NGRAM_N]> {
    let mut out = std::collections::HashSet::new();
    for s in streams {
        if stream_len_lt_ngram(s) {
            let mut g = [0u8; NGRAM_N];
            for (i, b) in s.iter().enumerate() {
                g[i] = *b;
            }
            out.insert(g);
            continue;
        }
        for w in s.windows(NGRAM_N) {
            let mut g = [0u8; NGRAM_N];
            g.copy_from_slice(w);
            out.insert(g);
        }
    }
    out
}

fn stream_len_lt_ngram(s: &[u8]) -> bool {
    s.len() < NGRAM_N
}

/// Action Jaccard distance `D_act = 1 - |A ∩ B| / |A ∪ B|`.
pub fn action_jaccard_distance(x: &[Stream], y: &[Stream]) -> f64 {
    let a = action_set(x);
    let b = action_set(y);
    let union: Vec<&[u8; NGRAM_N]> = a.union(&b).collect();
    if union.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(&b).count();
    1.0 - inter as f64 / union.len() as f64
}

/// Run the Gate 0 attestation.
pub fn gate0(prod: &[Stream], cand: &[Stream]) -> Gate0Result {
    let half = (prod.len() / 2).max(1);
    let pp = mmd2(&prod[..half], &prod[half..], SIGMA_BW);
    let eps_prod = pp.max(MMD2_FLOOR);

    let mmd2 = mmd2(cand, prod, SIGMA_BW);
    let d_act = action_jaccard_distance(cand, prod);
    let cosine = cosine_distance(&mean_embedding(cand), &mean_embedding(prod));

    let (passed, reason) = if mmd2 > eps_prod {
        (
            false,
            format!(
                "GATE0_FAIL: MMD^2={:.4} > eps_prod={:.4} (distribution shift / tail exfiltration)",
                mmd2, eps_prod
            ),
        )
    } else if d_act >= D_ACT_THRESHOLD {
        (
            false,
            format!("GATE0_FAIL: Action Jaccard D_act={:.4} >= 0.05", d_act),
        )
    } else {
        (true, "GATE0_PASS: distributionally aligned with production".to_string())
    };

    Gate0Result {
        mmd2,
        eps_prod,
        d_act,
        cosine_dist: cosine,
        passed,
        reason,
    }
}