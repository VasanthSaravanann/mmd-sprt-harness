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
    let (_x, _y, w) = assemble_combined(&px, &py, &pp, &ex, &ey, &ep);
    assert!(effective_sample_size(&w) >= N_EFF_MIN as f64);
}

#[test]
fn constants() {
    assert_eq!(N_EFF_MIN, 30);
    assert!((R_MIN - 0.85).abs() < 1e-9);
    assert!((P_MAX - 0.01).abs() < 1e-9);
}