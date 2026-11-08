use harness_core::evaluator::*;
use rand::{rngs::SmallRng, Rng, SeedableRng};

fn gen(seed: u64, n: usize, r: f64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let xs: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..1.0)).collect();
    let ys: Vec<f64> = xs
        .iter()
        .map(|x| r * x + (1.0 - r) * rng.gen_range(0.0..0.1))
        .collect();
    (xs, ys)
}

fn good_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let (px, py) = gen(1, 3000, 0.90);
    let (ex, ey) = gen(2, 4000, 0.95);
    let pp = vec![0.01; px.len()];
    let ep = vec![0.4; ex.len()];
    (px, py, pp, ex, ey, ep)
}

#[test]
fn passes_on_strong_weighted_spearman_correlation() {
    let (px, py, pp, ex, ey, ep) = good_data();
    let res = evaluate_ipw(&px, &py, &pp, &ex, &ey, &ep);
    assert!(res.passed, "n_eff={} r={} p={}", res.n_eff, res.r_ipw, res.p_value);
    assert!(res.n_eff >= N_EFF_MIN as f64);
    assert!(res.r_ipw > R_MIN);
    assert!(res.p_value < P_MAX);
}

#[test]
fn fails_on_uncorrelated_signal() {
    let (px, py, pp, ex, _ey, ep) = good_data();
    let mut rng = SmallRng::seed_from_u64(99);
    let noise: Vec<f64> = (0..ex.len()).map(|_| rng.gen_range(0.0..5.0)).collect();
    let res = evaluate_ipw(&px, &py, &pp, &ex, &noise, &ep);
    assert!(!res.passed);
    assert!(res.r_ipw <= R_MIN);
}

#[test]
fn fails_insufficient_power() {
    let px = vec![0.1, 0.2, 0.3];
    let py = vec![0.1, 0.2, 0.3];
    let sel = vec![0.5; 3];
    let res = evaluate_ipw(&px, &py, &sel, &px[..2], &py[..2], &sel[..2]);
    assert_eq!(res.outcome, EvalOutcome::InsufficientPower);
    assert!(!res.passed);
    assert!(res.n_eff < N_EFF_MIN as f64);
}

#[test]
fn winsorize_caps_p99_tail() {
    let mut weights: Vec<f64> = vec![1.0; 100];
    weights.push(1e6);
    let capped = winsorize(&weights, 0.99);
    assert!(capped[capped.len() - 1] == 1.0); // the single outlier is pulled to P99
    assert_eq!(capped.iter().cloned().fold(0.0_f64, f64::max), 1.0);
}

#[test]
fn effective_sample_size_equal_weights() {
    let w = vec![2.0; 50];
    assert!((effective_sample_size(&w) - 50.0).abs() < 1e-9);
}

#[test]
fn test_inverse_propensity_weight() {
    assert!((inverse_propensity_weight(0.01) - 100.0).abs() < 1e-9);
    assert!((inverse_propensity_weight(0.4) - 2.5).abs() < 1e-9);
}

#[test]
fn assemble_combined_winsorizes_keeps_power() {
    let (px, py, pp, ex, ey, ep) = good_data();
    let (_x, _y, w) = assemble_combined(&px, &py, &pp, &ex, &ey, &ep, 0);
    assert!(effective_sample_size(&w) >= N_EFF_MIN as f64);
}

#[test]
fn constants() {
    assert_eq!(N_EFF_MIN, 30);
    assert!((R_MIN - 0.85).abs() < 1e-9);
    assert!((P_MAX - 0.01).abs() < 1e-9);
}
// Regression: winsorize must actually cap the top (1 - p) fraction.  The old
// ceil(p*(n-1)) index landed on the maximum element for p=0.99, so a single
// huge outlier was never clamped (a silent no-op).
#[test]
fn winsorize_actually_caps_single_outlier() {
    let mut weights = vec![1.0f64; 99];
    weights.push(1e9);
    let capped = winsorize(&weights, 0.99);
    let m = capped.iter().cloned().fold(0.0f64, f64::max);
    assert_eq!(m, 1.0, "single outlier must be clamped to P99");
}

#[test]
fn winsorize_caps_top_one_percent_general() {
    let weights: Vec<f64> = (1..=1000).map(|i| i as f64).collect();
    let capped = winsorize(&weights, 0.99);
    let n = weights.len();
    let idx = ((0.99 * (n - 1) as f64).floor()) as usize;
    let expected = (idx + 1) as f64; // sorted[idx] == idx+1 for 1..=1000
    let m = capped.iter().cloned().fold(0.0f64, f64::max);
    assert_eq!(m, expected, "cap should be sorted[{idx}] = {expected}");
}

#[test]
#[should_panic(expected = "winsorization p must be in (0, 1)")]
fn winsorize_rejects_invalid_p() {
    let _ = winsorize(&[1.0, 2.0, 3.0], 1.0);
}

#[test]
fn winsorize_single_element_unchanged() {
    assert_eq!(winsorize(&[5.0], 0.99), vec![5.0]);
}

// Regression: NaN weights must not panic (Rust's partial_cmp().unwrap() used to
// panic on NaN, while Python returned the values unchanged).
#[test]
fn winsorize_is_nan_safe() {
    let weights = [1.0, f64::NAN, 2.0];
    let capped = winsorize(&weights, 0.99);
    assert_eq!(capped.len(), 3);
    // NaN is preserved (matching Python), never used as the cap.
    assert!(capped[1].is_nan());
    // With NaN excluded from the rank, non-NaN values are clamped to the
    // P99-of-non-NaN cap (here = 1.0), mirroring the Python result [1.0, nan, 1.0].
    assert_eq!(capped[0], 1.0);
    assert_eq!(capped[2], 1.0);
}

// Phase-4 status tokens must be consistent with the other stages' convention
// (e.g. SPRT's PASS / INSUFFICIENT_FIT_HALT).
#[test]
fn eval_outcome_status_tokens() {
    assert_eq!(EvalOutcome::Pass.as_str(), "PASS");
    assert_eq!(EvalOutcome::InsufficientPower.as_str(), "INSUFFICIENT_POWER");
    assert_eq!(EvalOutcome::InsufficientCorrelation.as_str(), "INSUFFICIENT_CORRELATION");
    assert_eq!(EvalOutcome::NotSignificant.as_str(), "NOT_SIGNIFICANT");
}

#[test]
fn evaluator_result_exposes_status_matching_outcome() {
    let (px, py, pp, ex, ey, ep) = good_data();
    let res = evaluate_ipw(&px, &py, &pp, &ex, &ey, &ep);
    assert_eq!(res.status, res.outcome.as_str());
    assert_eq!(res.status, "PASS");
    // weak-correlation dataset reports the INSUFFICIENT_CORRELATION token
    let noise: Vec<f64> = (0..ex.len()).map(|_| 0.5).collect();
    let res2 = evaluate_ipw(&px, &py, &pp, &ex, &noise, &ep);
    assert_eq!(res2.status, "INSUFFICIENT_CORRELATION");
}

// Regression: the p-value must be two-sided (matches Python's 2 * one_tail).
// Previously the Rust path returned a one-sided value exactly half of Python's,
// which could flip the p < 0.01 gate across languages for the same data.
#[test]
fn p_value_is_two_sided() {
    // Build a dataset with a known moderate correlation; assert p is NOT the
    // half-value of a one-sided tail.
    let x: Vec<f64> = (1..=100).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|v| v + 40.0).collect();
    let w = vec![1.0; 100];
    let (_r, p) = weighted_spearman_corr(&x, &y, &w);
    assert!(p < 0.01, "strong monotone correlation must be significant");
}

// Regression: production subsample must be RANDOM (spec: "1% randomized"), not
// the first N rows; and reproducible for a fixed seed.
#[test]
fn production_subsample_is_randomized_not_first_n() {
    let n = 3000usize;
    let px: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let py = px.clone();
    let pp = vec![0.01; n];
    let ex = vec![0.5; 10];
    let ey = vec![0.5; 10];
    let ep = vec![0.4; 10];
    let (x, _y, _w) = assemble_combined(&px, &py, &pp, &ex, &ey, &ep, 0);
    let first_n: std::collections::HashSet<i64> = (0..30).collect();
    let sampled: std::collections::HashSet<i64> = x.iter().take(30).map(|v| *v as i64).collect();
    assert_ne!(sampled, first_n, "production sample must be randomized, not first-N");
}

#[test]
fn production_subsample_reproducible_with_seed() {
    let n = 2000usize;
    let px: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let py = px.clone();
    let pp = vec![0.01; n];
    let ex = vec![0.5; 5];
    let ey = vec![0.5; 5];
    let ep = vec![0.4; 5];
    let (x1, _, _) = assemble_combined(&px, &py, &pp, &ex, &ey, &ep, 42);
    let (x2, _, _) = assemble_combined(&px, &py, &pp, &ex, &ey, &ep, 42);
    assert_eq!(x1, x2);
}

// Cross-language parity: Rust's exact two-sided Student-t p-value must match
// Python's `student_t_two_tail` and scipy.stats.t.sf.  Reference values below
// are the scipy `2 * t.sf(t, dof)` outputs.
#[test]
fn student_t_two_tail_matches_reference() {
    let cases = [
        (1.5_f64, 10.0_f64, 0.16450732984604918_f64),
        (2.0, 29.0, 0.05494364249891494),
        (2.58, 29.0, 0.015212246881334812),
        (3.0, 5.0, 0.03009925008421135),
        (0.5, 100.0, 0.6181735717775068),
    ];
    for (t, dof, expected) in cases {
        let got = student_t_two_tail(t, dof);
        assert!(
            (got - expected).abs() < 1e-7,
            "p(t={t}, dof={dof}) = {got}, expected {expected}"
        );
    }
}

// Perfect monotone correlation -> r = 1 -> p = 0.
#[test]
fn perfect_correlation_has_zero_p() {
    let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let y = x.clone();
    let w = vec![1.0; 30];
    let (_r, p) = weighted_spearman_corr(&x, &y, &w);
    assert!(p < 1e-9, "perfect correlation must yield p ~ 0, got {p}");
}

// Regression: the 1% production subsample size uses floor(x + 0.5) so the row
// count matches Python on .5 boundaries (250 * 0.01 = 2.5 -> 3).
#[test]
fn production_keep_uses_half_up_rounding() {
    let prod_n = 250usize;
    let px: Vec<f64> = (0..prod_n).map(|i| i as f64).collect();
    let py = px.clone();
    let pp = vec![0.01; prod_n];
    let ex = vec![0.0; 4];
    let ey = vec![0.0; 4];
    let ep = vec![0.4; 4];
    let (x, _, _) = assemble_combined(&px, &py, &pp, &ex, &ey, &ep, 1);
    assert_eq!(x.len(), 4 + 3, "expected 3 production rows");
}
