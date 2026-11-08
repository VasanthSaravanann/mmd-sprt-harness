//! Phase-4 gate: IPW correlation evaluator.
//!
//! Combines a 1% randomized production sample with a stratified, high-"perf"
//! evaluation sample.  Inverse-propensity weights are winsorized at the 99th
//! percentile to bound variance and enforce `n_eff >= 30`.
//!
//! The gate PASSES iff `n_eff >= 30` AND the weighted **Spearman** correlation
//! `r_{2,IPW} > 0.85`; datasets with insufficient power fail as
//! `INSUFFICIENT_POWER` and correlated-but-weak ones fail as `GATE0_FAIL` /
//! `DEPLOYMENT_HALTED`.

pub const N_EFF_MIN: usize = 30;
pub const R_MIN: f64 = 0.85;
pub const P_MAX: f64 = 0.01;
pub const PROD_FRACTION: f64 = 0.01;

/// Outcome of the evaluation gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalOutcome {
    Pass,
    InsufficientPower,
    WeakCorrelation,
    NotSignificant,
}

/// Result of the IPW evaluator.
pub struct EvaluatorResult {
    pub n_eff: f64,
    pub r_ipw: f64,
    pub p_value: f64,
    pub outcome: EvalOutcome,
    pub passed: bool,
}

fn propensity_probability(select_prob: f64) -> f64 {
    select_prob.clamp(1e-9, 1.0)
}

pub fn inverse_propensity_weight(select_prob: f64) -> f64 {
    1.0 / propensity_probability(select_prob)
}

/// Cap weights at the `p`-th percentile (P99 by default).
pub fn winsorize(weights: &[f64], p: f64) -> Vec<f64> {
    if weights.is_empty() {
        return Vec::new();
    }
    let mut sorted: Vec<f64> = weights.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((p * (sorted.len() - 1) as f64).ceil() as usize).min(sorted.len() - 1);
    let cap = sorted[idx];
    weights.iter().map(|w| w.min(cap)).collect()
}

/// Kish's effective sample size `n_eff = (Σw)² / Σw²`.
pub fn effective_sample_size(weights: &[f64]) -> f64 {
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let square_sum: f64 = weights.iter().map(|w| w * w).sum();
    total * total / square_sum
}

/// Rank ties avenged average ranks (used for Spearman).
fn average_ranks(data: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..data.len()).collect();
    order.sort_by(|a, b| data[*a].partial_cmp(&data[*b]).unwrap());
    let mut ranks = vec![0.0f64; data.len()];
    let mut i = 0usize;
    while i < order.len() {
        let mut j = i;
        let val = data[order[i]];
        while j + 1 < order.len() && data[order[j + 1]] == val {
            j += 1;
        }
        let avg = ((i + j) as f64) / 2.0 + 1.0;
        for k in order[i..=j].iter() {
            ranks[*k] = avg;
        }
        i = j + 1;
    }
    ranks
}

/// Standard-normal survival (two-sided p for the correlation t-test).
fn norm_two_tail(z: f64) -> f64 {
    0.5 * (1.0 - erf(z / 2f64.sqrt()))
}

fn erf(x: f64) -> f64 {
    // Abramowitz-Stegun 7.1.26 approximation.
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    sign * (1.0 - poly * (-x * x).exp())
}

/// Weighted Spearman (rank-based IPW) correlation `(r, p)`.
pub fn weighted_spearman_corr(x: &[f64], y: &[f64], weights: &[f64]) -> (f64, f64) {
    let n = x.len();
    assert!(n == y.len() && n == weights.len() && n >= 3);
    let _ = n;
    let rx = average_ranks(x);
    let ry = average_ranks(y);
    weighted_pearson_corr(&rx, &ry, weights)
}

/// Weighted Pearson correlation on (already-ranked) inputs.
pub fn weighted_pearson_corr(x: &[f64], y: &[f64], weights: &[f64]) -> (f64, f64) {
    assert!(x.len() == y.len() && x.len() == weights.len() && x.len() >= 3);
    let wsum: f64 = weights.iter().sum();
    if wsum <= 0.0 {
        return (0.0, 1.0);
    }
    let mx: f64 = x.iter().zip(weights).map(|(xi, wi)| xi * wi).sum::<f64>() / wsum;
    let my: f64 = y.iter().zip(weights).map(|(yi, wi)| yi * wi).sum::<f64>() / wsum;
    let cov: f64 = x
        .iter()
        .zip(y)
        .zip(weights)
        .map(|((xi, yi), wi)| wi * (xi - mx) * (yi - my))
        .sum::<f64>()
        / wsum;
    let vx: f64 = x.iter().zip(weights).map(|(xi, wi)| wi * (xi - mx).powi(2)).sum::<f64>() / wsum;
    let vy: f64 = y.iter().zip(weights).map(|(yi, wi)| wi * (yi - my).powi(2)).sum::<f64>() / wsum;
    let denom = (vx * vy).sqrt();
    if denom == 0.0 {
        return (0.0, 1.0);
    }
    let r = cov / denom;
    if r.abs() >= 1.0 {
        return (r, 0.0);
    }
    // t-test with n_eff - 1 degrees of freedom, normal-approximated tail.
    let n_eff = effective_sample_size(weights);
    let dof = (n_eff - 1.0).max(1.0);
    let t = r * (dof / (1.0 - r * r)).sqrt();
    let p = norm_two_tail(t.abs());
    (r, p)
}

/// Combine 1%-production + stratified eval samples, then winsorize IPW weights.
pub fn assemble_combined(
    prod_x: &[f64],
    prod_y: &[f64],
    prod_sel: &[f64],
    eval_x: &[f64],
    eval_y: &[f64],
    eval_sel: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut w = Vec::new();

    let keep = ((prod_x.len() as f64) * PROD_FRACTION).round() as usize;
    let keep = keep.clamp(1, prod_x.len());
    for i in 0..keep {
        x.push(prod_x[i]);
        y.push(prod_y[i]);
        w.push(inverse_propensity_weight(prod_sel[i]));
    }
    for i in 0..eval_x.len() {
        x.push(eval_x[i]);
        y.push(eval_y[i]);
        w.push(inverse_propensity_weight(eval_sel[i]));
    }
    let w = winsorize(&w, 0.99);
    (x, y, w)
}

/// Run the Phase-4 IPW correlation gate.
///
/// Returns `(x, y, weights)` plus a summarized result in a tuple
/// `(passed, n_eff, r, p, reason)`.
pub fn evaluate_ipw_vec(
    prod_x: &[f64],
    prod_y: &[f64],
    prod_sel: &[f64],
    eval_x: &[f64],
    eval_y: &[f64],
    eval_sel: &[f64],
) -> Vec<f64> {
    let (x, y, w) = assemble_combined(prod_x, prod_y, prod_sel, eval_x, eval_y, eval_sel);
    let n_eff = effective_sample_size(&w);
    let (r, p) = weighted_spearman_corr(&x, &y, &w);
    let (passed, _outcome) = if n_eff < N_EFF_MIN as f64 {
        (false, EvalOutcome::InsufficientPower)
    } else if r <= R_MIN {
        (false, EvalOutcome::WeakCorrelation)
    } else if p >= P_MAX {
        (false, EvalOutcome::NotSignificant)
    } else {
        (true, EvalOutcome::Pass)
    };
    vec![if passed { 1.0 } else { 0.0 }, n_eff, r, p]
}

/// Structured result form of the IPW gate.
pub fn evaluate_ipw(
    prod_x: &[f64],
    prod_y: &[f64],
    prod_sel: &[f64],
    eval_x: &[f64],
    eval_y: &[f64],
    eval_sel: &[f64],
) -> EvaluatorResult {
    let (x, y, w) = assemble_combined(prod_x, prod_y, prod_sel, eval_x, eval_y, eval_sel);
    let n_eff = effective_sample_size(&w);
    let (r, p) = weighted_spearman_corr(&x, &y, &w);
    let (passed, outcome) = if n_eff < N_EFF_MIN as f64 {
        (false, EvalOutcome::InsufficientPower)
    } else if r <= R_MIN {
        (false, EvalOutcome::WeakCorrelation)
    } else if p >= P_MAX {
        (false, EvalOutcome::NotSignificant)
    } else {
        (true, EvalOutcome::Pass)
    };
    EvaluatorResult {
        n_eff,
        r_ipw: r,
        p_value: p,
        outcome,
        passed,
    }
}