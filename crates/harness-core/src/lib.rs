//! `harness-core` — the mathematical kernels of the distributional-attestation,
//! SPRT, and IPW-evaluation pipeline, ported to Rust.
//!
//! * `gate0`     — length-normalized 3-gram spectrum, Gaussian-RBF MMD² with the
//!                production floor, and Action Jaccard distance.
//! * `sprt`      — parallel fuzzing + N≥5 rolling exponential-decay convergence
//!                diagnostics (PASS / INSUFFICIENT_FIT_HALT / BUDGET_EXHAUSTED).
//! * `evaluator` — 1% production + stratified high-“perf” IPW evaluator with
//!                P99 winsorization and power / correlation gates.

pub mod evaluator;
pub mod gate0;
pub mod sprt;

pub use gate0::{gate0, Gate0Result};
pub use sprt::{assess_convergence, fit_exponential_decay, DecayFit, Verdict};

/// Common precision helpers used across the kernels.
pub mod prelude {
    pub const MMD2_FLOOR: f64 = 0.015;
    pub const D_ACT_THRESHOLD: f64 = 0.05;
}