//! Prior-anchored SPRT iteration loop.
//!
//! Runs a fixed-point fuzzing loop (up to `k_max = 6` rounds), each mutating a
//! stream `K ∈ [3, 8]` times.  Viability is scored via a Wilson 95% CI and the
//! model's uncertainty is captured by the binary logit entropy.
//!
//! Convergence is judged by fitting the exponential decay
//! `d^(i) = beta_0 * exp(-lambda * i)` over a rolling `N >= 5` window.  We index
//! iterations from 0 and extrapolate the fitted model to iteration `i = 6`.
//!
//! * `R^2 >= 0.90` and `d^(6) < 0.02` -> `PASS`
//! * `R^2 < 0.90` -> `INSUFFICIENT_FIT_HALT`
//! * otherwise (not enough data) -> `BUDGET_EXHAUSTED_CONVERGING`

use rayon::prelude::*;

pub const K_MAX: usize = 6;
pub const K_MUT_MIN: usize = 3;
pub const K_MUT_MAX: usize = 8;
pub const WINDOW_MIN: usize = 5;
pub const R2_THRESHOLD: f64 = 0.90;
pub const D6_THRESHOLD: f64 = 0.02;
const Z_95: f64 = 1.959_963_984_540_054;

/// Outcome of the convergence diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    InsufficientFitHalt,
    BudgetExhaustedConverging,
}

impl Verdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            Verdict::Pass => "PASS",
            Verdict::InsufficientFitHalt => "INSUFFICIENT_FIT_HALT",
            Verdict::BudgetExhaustedConverging => "BUDGET_EXHAUSTED_CONVERGING",
        }
    }
}

/// Result of fitting the exponential decay.
pub struct DecayFit {
    pub beta_0: f64,
    pub lam: f64,
    pub r2: f64,
    /// Extrapolated distance at iteration `i = 6`.
    pub d6: f64,
    pub rd_verdict: Verdict,
    pub converged: bool,
}

/// 95% Wilson score interval `(lo, hi)` for a binomial proportion.
pub fn wilson_ci(successes: usize, trials: usize) -> (f64, f64) {
    if trials == 0 {
        return (0.0, 0.0);
    }
    let p = successes as f64 / trials as f64;
    let z2 = Z_95 * Z_95;
    let denom = 1.0 + z2 / trials as f64;
    let center = (p + z2 / (2.0 * trials as f64)) / denom;
    let margin =
        Z_95 * (((p * (1.0 - p) + z2 / (4.0 * trials as f64)) / trials as f64).sqrt()) / denom;
    (center - margin, center + margin)
}

/// Binary logit entropy `H = -p log p - (1-p) log(1-p)`.
pub fn logit_entropy(prob: f64) -> f64 {
    let eps = 1e-9f64;
    let p = prob.clamp(eps, 1.0 - eps);
    -(p * p.ln() + (1.0 - p) * (1.0 - p).ln())
}

/// Fit `d^(i) = beta_0 * exp(-lambda * i)` over a rolling `N >= 5` window.
///
/// Points are indexed from `i = 0..N-1`; `d6` is the extrapolated value at
/// iteration `i = 6`.  Requires strictly positive distances.
pub fn fit_exponential_decay(distances: &[f64]) -> DecayFit {
    let n = distances.len();
    assert!(
        n >= WINDOW_MIN,
        "exponential decay requires at least {} points, got {}",
        WINDOW_MIN,
        n
    );
    assert!(
        distances.iter().all(|v| *v > 0.0),
        "exponential decay requires strictly positive distances"
    );

    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = distances.iter().map(|v| v.ln()).collect();
    let xbar = x.iter().sum::<f64>() / n as f64;
    let ybar = y.iter().sum::<f64>() / n as f64;
    let sxx: f64 = x.iter().map(|xi| (xi - xbar).powi(2)).sum();
    let sxy: f64 = x.iter().zip(&y).map(|(xi, yi)| (xi - xbar) * (yi - ybar)).sum();
    let slope = if sxx.abs() > 1e-12 { sxy / sxx } else { 0.0 };
    let intercept = ybar - slope * xbar;

    let lam = -slope;
    let beta_0 = intercept.exp();
    let d_pred: Vec<f64> = x.iter().map(|i| beta_0 * (-lam * i).exp()).collect();

    let ymean = distances.iter().sum::<f64>() / n as f64;
    let ss_res: f64 = distances
        .iter()
        .zip(&d_pred)
        .map(|(vi, pi)| (vi - pi).powi(2))
        .sum();
    let ss_tot: f64 = distances.iter().map(|vi| (vi - ymean).powi(2)).sum();
    let r2 = if ss_tot > 1e-12 { 1.0 - ss_res / ss_tot } else { 1.0 };

    let d6 = beta_0 * (-lam * 6.0).exp();

    let (verdict, converged) = if r2 < R2_THRESHOLD {
        (Verdict::InsufficientFitHalt, false)
    } else if d6 < D6_THRESHOLD {
        (Verdict::Pass, true)
    } else {
        (Verdict::BudgetExhaustedConverging, false)
    };

    DecayFit {
        beta_0,
        lam,
        r2,
        d6,
        rd_verdict: verdict,
        converged,
    }
}

/// Convenience wrapper over [`fit_exponential_decay`].
pub fn assess_convergence(distances: &[f64]) -> DecayFit {
    fit_exponential_decay(distances)
}

// --------------------------------------------------------------------------
// Parallel fuzzer + SPRT loop
// --------------------------------------------------------------------------

/// A single fuzzing rollout over a stream mutation (e.g., Base64 / Hex / prompt
/// injection).  Returns the post-mutation distance for the round.
pub type FuzzerFn = dyn Fn(usize, usize) -> f64 + Send + Sync;

/// Outcome of the full loop.
pub struct SprtLoopResult {
    pub distances: Vec<f64>,
    pub k_used: usize,
    pub mutation_counts: Vec<usize>,
    pub viability: Vec<f64>,
    pub viab_lower: Vec<f64>,
    pub viab_upper: Vec<f64>,
    pub h_fuzz: Vec<f64>,
    pub fit: Option<DecayFit>,
}

/// Simple deterministic LCG for reproducible parallelism tests.
pub struct Lcg(pub u64);
impl Lcg {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    pub fn in_range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next() as usize) % (hi - lo + 1)
    }
}

/// Run a parallel SPRT loop over `k_max` rounds.
///
/// `fuzzer(round_index, worker_idx)` returns the mutant distance `d^(i)` for
/// round `i`.  Each round fans out `k` independent rollout workers via rayon.
pub fn run_sprt_loop(
    fuzzer: &FuzzerFn,
    k_max: usize,
    k_mut_min: usize,
    k_mut_max: usize,
    seed: u64,
) -> SprtLoopResult {
    let mut rng = Lcg(seed.wrapping_add(0x9e3779b97f4a7c15));
    let mut distances = Vec::with_capacity(k_max);
    let mut mutation_counts = Vec::with_capacity(k_max);
    let mut viability = Vec::new();
    let mut viab_lower = Vec::new();
    let mut viab_upper = Vec::new();
    let mut h_fuzz = Vec::new();
    let mut fit: Option<DecayFit> = None;

    for i in 0..k_max {
        let k = rng.in_range(k_mut_min, k_mut_max);
        mutation_counts.push(k);

        // Parallel rollout of the k mutations for this round.
        let d_i: f64 = (0..k).into_par_iter().map(|worker| fuzzer(i, worker)).sum::<f64>() / k as f64;
        distances.push(d_i);

        // Viability from this round (simulated successes/trials).
        let s = rng.next() as usize % 6;
        let t = 5usize;
        let (lo, hi) = wilson_ci(s, t);
        let point = s as f64 / t as f64;
        viability.push(point);
        viab_lower.push(lo);
        viab_upper.push(hi);
        h_fuzz.push(logit_entropy(point));

        if distances.len() >= WINDOW_MIN {
            fit = Some(fit_exponential_decay(&distances));
        }
    }

    SprtLoopResult {
        k_used: distances.len(),
        distances,
        mutation_counts,
        viability,
        viab_lower,
        viab_upper,
        h_fuzz,
        fit,
    }
}