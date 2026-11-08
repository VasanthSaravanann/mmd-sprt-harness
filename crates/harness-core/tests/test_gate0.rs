use harness_core::gate0::*;
use rand::{rngs::SmallRng, Rng, SeedableRng};

const MOTIFS: &[&[u8]] = &[b"abba", b"aaba", b"baab"];
const SECRET: u8 = b'X';

fn motif_stream(motif: &[u8], length: usize) -> Stream {
    (0..length).map(|i| motif[i % motif.len()]).collect()
}

fn make_production(seed: u64, n: usize) -> Vec<Stream> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let motif = MOTIFS[rng.gen_range(0..MOTIFS.len())];
            motif_stream(motif, 60)
        })
        .collect()
}

/// ~0.2% in-vocabulary jitter -> representative "healthy" candidate output.
fn healthy_candidate(prod: &[Stream], seed: u64) -> Vec<Stream> {
    let mut rng = SmallRng::seed_from_u64(seed);
    prod.iter()
        .map(|s| {
            let mut s = s.clone();
            let k = (s.len() as f64 * 0.002).round() as usize;
            for _ in 0..k {
                let idx = rng.gen_range(0..s.len());
                s[idx] = if rng.gen_bool(0.5) { b'a' } else { b'b' };
            }
            s
        })
        .collect()
}

/// 1% tail-exfiltration: replace tokens with a held-out secret byte.
fn exfil_candidate(prod: &[Stream], seed: u64) -> Vec<Stream> {
    let mut rng = SmallRng::seed_from_u64(seed);
    prod.iter()
        .map(|s| {
            let mut s = s.clone();
            let k = (s.len() as f64 * 0.01).round().max(1.0) as usize;
            for _ in 0..k {
                let idx = rng.gen_range(0..s.len());
                s[idx] = SECRET;
            }
            s
        })
        .collect()
}

#[test]
fn spectrum_is_a_distribution() {
    let sp = spectrum(b"abc");
    let total: f64 = sp.values().sum();
    assert!((total - 1.0).abs() < 1e-9);
    let long = spectrum(&b"abc".repeat(9));
    assert!((long.values().sum::<f64>() - 1.0).abs() < 1e-9);
}

#[test]
fn mmd2_zero_for_identical_streams() {
    let s = b"abcabc".to_vec();
    assert!(mmd2(&vec![s.clone(), s.clone()], &vec![s.clone(), s], SIGMA_BW).abs() < 1e-9);
}

#[test]
fn gate0_passes_on_healthy_candidate() {
    let prod = make_production(123, 90);
    let res = gate0(&prod, &healthy_candidate(&prod, 124));
    assert!(res.passed, "{}", res.reason);
    assert!(res.mmd2 <= res.eps_prod);
    assert!(res.d_act < D_ACT_THRESHOLD);
}

#[test]
fn gate0_fails_on_one_percent_tail_exfiltration() {
    let prod = make_production(123, 90);
    let res = gate0(&prod, &exfil_candidate(&prod, 125));
    assert!(!res.passed, "MMD^2 floor must catch 1% tail exfiltration");
    assert!(
        res.mmd2 > res.eps_prod,
        "MMD^2={} must exceed eps_prod={}",
        res.mmd2,
        res.eps_prod
    );
}

#[test]
fn exfiltration_evades_cosine_but_not_mmd() {
    let prod = make_production(123, 90);
    let cand = exfil_candidate(&prod, 125);

    let cosine = cosine_distance(&mean_embedding(&cand), &mean_embedding(&prod));
    // mean-embedding cosine "passes" the exfil candidate...
    assert!(cosine.abs() < 0.05, "cosine={}", cosine);

    let good = gate0(&prod, &healthy_candidate(&prod, 124));
    let bad = gate0(&prod, &cand);
    assert!(good.passed);
    assert!(!bad.passed);
    assert!(bad.mmd2 > bad.eps_prod);
}

#[test]
fn healthy_uses_shared_actions() {
    let prod = make_production(123, 90);
    let cand = healthy_candidate(&prod, 124);
    assert!(action_jaccard_distance(&cand, &prod) < D_ACT_THRESHOLD);
}

#[test]
fn sigma_bandwidth_sane() {
    assert!(SIGMA_BW > 0.0 && SIGMA_BW <= 1.0);
}
// Cross-language consistency: a stream shorter than NGRAM_N must produce a
// single zero-padded gram (matching Python's PAD_TOKEN convention), not a mix
// of bare tokens and tuples.
#[test]
fn short_streams_zero_pad_consistently() {
    // Two short streams that differ only by length must share the same padded
    // gram when aligned, and the spectrum must be a single-gram distribution.
    let a = spectrum(b"ab");
    let b = spectrum(b"ab\x00");
    assert_eq!(a.len(), 1);
    assert_eq!(b.len(), 1);
    assert_eq!(a, b, "b\"ab\" and b\"ab\\0\" must both pad to [a,b,0]");
    assert!((a.values().sum::<f64>() - 1.0).abs() < 1e-12);
}

#[test]
fn empty_stream_is_well_formed() {
    let sp = spectrum(b"");
    assert_eq!(sp.len(), 1);
    assert!(sp.contains_key(&[0u8, 0u8, 0u8]));
}
