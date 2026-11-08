//! Phase-4 gate: IPW correlation evaluator.
//!
//! Combines a 1% randomized production sample with a stratified, high-"perf"
//! evaluation sample.  Inverse-propensity weights are winsorized at the 99th
//! percentile to bound variance and enforce `n_eff >= 30`.
//!
//! The gate PASSES iff `n_eff >= 30` AND the weighted **Spearman** correlation
//! `r_{2,IPW} > 0.85` with `p < 0.01`.  It fails with the status tokens
//! `INSUFFICIENT_POWER`, `INSUFFICIENT_CORRELATION` or `NOT_SIGNIFICANT`,
//! mirroring the uniform status-string convention used by the Gate 0 and SPRT
//! stages (e.g. `GATE0_PASS`, `INSUFFICIENT_FIT_HALT`).

pub const N_EFF_MIN: usize = 30;
pub const R_MIN: f64 = 0.85;
pub const P_MAX: f64 = 0.01;
pub const PROD_FRACTION: f64 = 0.01;

/// Outcome of the evaluation gate.
///
/// The variant is reported as a uniform status string via [`EvalOutcome::as_str`]
/// so every pipeline stage exposes a consistent verdict token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalOutcome {
    Pass,
    InsufficientPower,
    InsufficientCorrelation,
    NotSignificant,
}

impl EvalOutcome {
    /// Canonical status token for this outcome.
    pub fn as_str(&self) -> &'static str {
        match self {
            EvalOutcome::Pass => "PASS",
            EvalOutcome::InsufficientPower => "INSUFFICIENT_POWER",
            EvalOutcome::InsufficientCorrelation => "INSUFFICIENT_CORRELATION",
            EvalOutcome::NotSignificant => "NOT_SIGNIFICANT",
        }
    }
}

/// Result of the IPW evaluator.
pub struct EvaluatorResult {
    pub n_eff: f64,
    pub r_ipw: f64,
    pub p_value: f64,
    pub outcome: EvalOutcome,
    /// Canonical status token: `PASS`, `INSUFFICIENT_POWER`,
    /// `INSUFFICIENT_CORRELATION` or `NOT_SIGNIFICANT`.
    pub status: &'static str,
    pub passed: bool,
}

fn propensity_probability(select_prob: f64) -> f64 {
    select_prob.clamp(1e-9, 1.0)
}

pub fn inverse_propensity_weight(select_prob: f64) -> f64 {
    1.0 / propensity_probability(select_prob)
}

/// Cap weights at the `p`-th percentile (P99 by default).
///
/// Uses the lower percentile rank `floor(p * (n - 1))` so the top `(1 - p)`
/// fraction is genuinely clamped.  `ceil` would land on the maximum element for
/// common `p` (e.g. `p=0.99, n=100` -> index 99) and silently leave outliers.
pub fn winsorize(weights: &[f64], p: f64) -> Vec<f64> {
    if weights.is_empty() {
        return Vec::new();
    }
    assert!(p > 0.0 && p < 1.0, "winsorization p must be in (0, 1), got {p}");
    let n = weights.len();
    if n == 1 {
        return weights.to_vec();
    }
    // NaN-safe total ordering.  `partial_cmp(..).unwrap()` panics on NaN
    // weights, whereas the Python implementation returns them unchanged.  We
    // sort NaNs to the end (total_cmp fallback) and skip them when choosing the
    // cap so a NaN never becomes the cap value.
    let mut sorted: Vec<f64> = weights.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or_else(|| a.total_cmp(b)));
    let non_nan = sorted.iter().filter(|v| !v.is_nan()).count();
    let idx = if non_nan == 0 {
        0
    } else {
        (((p * (non_nan - 1) as f64).floor()) as usize).min(non_nan - 1)
    };
    let cap = sorted[idx];
    // Preserve NaN entries (min(NaN, cap) in Rust would return cap and silently
    // replace them).
    weights.iter().map(|w| if w.is_nan() { *w } else { w.min(cap) }).collect()
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

/// Average ranks with NaN-safe ordering (used for Spearman).
fn average_ranks(data: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..data.len()).collect();
    order.sort_by(|a, b| {
        data[*a].partial_cmp(&data[*b]).unwrap_or_else(|| data[*a].total_cmp(&data[*b]))
    });
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

/// Natural log of the gamma function (Lanczos approximation, g = 7, n = 9).
fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // reflection formula
        std::f64::consts::PI.ln()
            - (std::f64::consts::PI * x).sin().abs().ln()
            - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        for (i, &c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        let t = x + G + 0.5;
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Continued-fraction evaluation of the incomplete beta function.
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAXIT: usize = 200;
    const EPS: f64 = 3.0e-16;
    const FPMIN: f64 = 1.0e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAXIT {
        let m2 = 2 * m;
        let m_f = m as f64;
        let mut aa = m_f * (b - m_f) * x / ((qam + m2 as f64) * (a + m2 as f64));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m_f) * (qab + m_f) * x / ((a + m2 as f64) * (qap + m2 as f64));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Regularized incomplete beta function `I_x(a, b)`.
fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// Exact two-sided Student-t p-value via the regularized incomplete beta:
/// `p = I_{dof/(dof + t^2)}(dof/2, 1/2)`.
///
/// Matches the Python `student_t_two_tail` exactly (the previous Rust version
/// used a normal approximation and an earlier revision even returned a
/// one-sided value, either of which could flip the `p < 0.01` gate across
/// languages).
pub fn student_t_two_tail(t: f64, dof: f64) -> f64 {
    if dof <= 0.0 {
        return 1.0;
    }
    let x = dof / (dof + t * t);
    betai(dof / 2.0, 0.5, x)
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
    let p = student_t_two_tail(t.abs(), dof);
    (r, p)
}

/// Combine a 1% **uniformly random** production subsample with the stratified
/// high-perf evaluation sample, then winsorize the IPW weights at P99.
///
/// The previous implementation took the first `keep` production rows — a
/// deterministic, order-biased subset contradicting the spec's "1% randomized
/// production data".  We now draw `keep` distinct indices via a seeded
/// Fisher-Yates shuffle (deterministic per `seed`).
pub fn assemble_combined(
    prod_x: &[f64],
    prod_y: &[f64],
    prod_sel: &[f64],
    eval_x: &[f64],
    eval_y: &[f64],
    eval_sel: &[f64],
    seed: u64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut w = Vec::new();

    // floor(x + 0.5) rather than `f64::round`'s half-away-from-zero, to match
    // the Python implementation exactly on .5 boundaries.
    let keep = ((prod_x.len() as f64) * PROD_FRACTION + 0.5).floor() as usize;
    let keep = keep.clamp(1, prod_x.len());
    // Deterministic seeded partial Fisher-Yates: sample `keep` distinct indices.
    let mut indices: Vec<usize> = (0..prod_x.len()).collect();
    let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(0x2545_F491_4F6C_DD1D);
    for i in 0..keep {
        // xorshift64*
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        let r = rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        let j = i + (r as usize) % (indices.len() - i);
        indices.swap(i, j);
    }
    for &i in &indices[..keep] {
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

/// Structured result form of the IPW gate.
///
/// Uses `seed = 0` for the production subsample; see [`evaluate_ipw_seeded`].
pub fn evaluate_ipw(
    prod_x: &[f64],
    prod_y: &[f64],
    prod_sel: &[f64],
    eval_x: &[f64],
    eval_y: &[f64],
    eval_sel: &[f64],
) -> EvaluatorResult {
    evaluate_ipw_seeded(prod_x, prod_y, prod_sel, eval_x, eval_y, eval_sel, 0)
}

/// Structured result form of the IPW gate with a caller-chosen sampling seed.
pub fn evaluate_ipw_seeded(
    prod_x: &[f64],
    prod_y: &[f64],
    prod_sel: &[f64],
    eval_x: &[f64],
    eval_y: &[f64],
    eval_sel: &[f64],
    seed: u64,
) -> EvaluatorResult {
    let (x, y, w) = assemble_combined(prod_x, prod_y, prod_sel, eval_x, eval_y, eval_sel, seed);
    let n_eff = effective_sample_size(&w);
    let (r, p) = weighted_spearman_corr(&x, &y, &w);
    let (passed, outcome) = if n_eff < N_EFF_MIN as f64 {
        (false, EvalOutcome::InsufficientPower)
    } else if r <= R_MIN {
        (false, EvalOutcome::InsufficientCorrelation)
    } else if p >= P_MAX {
        (false, EvalOutcome::NotSignificant)
    } else {
        (true, EvalOutcome::Pass)
    };
    let status = outcome.as_str();
    EvaluatorResult {
        n_eff,
        r_ipw: r,
        p_value: p,
        outcome,
        status,
        passed,
    }
}