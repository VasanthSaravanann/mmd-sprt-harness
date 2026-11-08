use harness_core::sprt::*;

#[test]
fn reference_pass_sequence() {
    // Clean exponential decay => R^2 ~ 0.99, extrapolated d^(6) < 0.02.
    let fit = assess_convergence(&[0.30, 0.20, 0.12, 0.07, 0.04]);
    assert_eq!(fit.rd_verdict, Verdict::Pass);
    assert!(fit.converged);
    assert!(fit.r2 >= R2_THRESHOLD);
    assert!(fit.d6 < D6_THRESHOLD);
}

#[test]
fn reference_halt_sequence() {
    // Jagged series => poor fit => INSUFFICIENT_FIT_HALT.
    let fit = assess_convergence(&[0.30, 0.10, 0.28, 0.05, 0.22]);
    assert_eq!(fit.rd_verdict, Verdict::InsufficientFitHalt);
    assert!(!fit.converged);
    assert!(fit.r2 < R2_THRESHOLD);
}

// Short windows no longer panic: they return a NEEDS_MORE_DATA fit, matching
// the Python implementation.
#[test]
fn short_window_reports_needs_more_data() {
    let fit = assess_convergence(&[0.5, 0.4, 0.3, 0.2]);
    assert_eq!(fit.rd_verdict, Verdict::NeedsMoreData);
    assert!(!fit.converged);
    assert!(fit.r2.is_nan());
    assert_eq!(Verdict::NeedsMoreData.as_str(), "NEEDS_MORE_DATA");
}

#[test]
fn nonpositive_distance_still_panics() {
    let r = std::panic::catch_unwind(|| assess_convergence(&[0.5, 0.4, 0.0, 0.2, 0.1]));
    assert!(r.is_err(), "non-positive distances must be rejected");
}

#[test]
fn fits_exact_exponential() {
    let d: Vec<f64> = (0..WINDOW_MIN).map(|i| (-0.5 * i as f64).exp()).collect();
    let fit = fit_exponential_decay(&d);
    assert!((fit.beta_0 - 1.0).abs() < 0.05);
    assert!((fit.lam - 0.5).abs() < 0.05);
    assert!(fit.r2 >= 0.999);
}

#[test]
fn wilson_ci_bounds() {
    let (lo, hi) = wilson_ci(10, 10);
    assert!(lo >= 0.0 && lo <= hi && hi <= 1.0);
    assert!(hi < 1.0, "95% CI widens below 1");
}

#[test]
fn wilson_ci_contains_point_estimate() {
    let (lo, hi) = wilson_ci(4, 10);
    assert!(lo <= 0.4 && 0.4 <= hi);
}

#[test]
fn logit_entropy_extremes_and_center() {
    assert!(logit_entropy(0.0).abs() < 1e-6);
    assert!(logit_entropy(1.0).abs() < 1e-6);
    let mid = logit_entropy(0.5);
    assert!((mid - 2f64.ln()).abs() < 1e-6);
    assert!(mid > logit_entropy(0.2) && mid > logit_entropy(0.9));
}

#[test]
fn run_sprt_loop_reaches_kmax() {
    let distances = |i: usize, _w: usize| 0.3f64.powi((i + 1) as i32);
    let res = run_sprt_loop(&distances, K_MAX, K_MUT_MIN, K_MUT_MAX, 7);
    assert_eq!(res.k_used, K_MAX);
    assert_eq!(res.distances.len(), K_MAX);
    for k in &res.mutation_counts {
        assert!((K_MUT_MIN..=K_MUT_MAX).contains(k));
    }
    assert_eq!(res.viability.len(), K_MAX);
    for (i, (lo, hi)) in res.viab_lower.iter().zip(&res.viab_upper).enumerate() {
        assert!(lo <= hi && *lo >= 0.0 && *hi <= 1.0, "round {i}");
    }
    assert!(res.fit.is_some());
}

#[test]
fn verdict_display() {
    assert_eq!(Verdict::Pass.as_str(), "PASS");
    assert_eq!(Verdict::InsufficientFitHalt.as_str(), "INSUFFICIENT_FIT_HALT");
}