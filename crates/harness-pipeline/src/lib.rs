//! Unified end-to-end harness pipeline.
//!
//! Wires Gate 0 (distributional attestation), the parallel SPRT loop, and the
//! IPW Phase-4 evaluator into a single pipeline.  In production this library is
//! fed events from an eBPF ringbuffer; here the ingestion boundary is the
//! `Ingestion` trait so the kernel-backed path can be dropped in later.

use harness_core::evaluator::{evaluate_ipw, EvaluatorResult};
use harness_core::gate0::{gate0, Gate0Result, Stream};
use harness_core::sprt::{
    assess_convergence, run_sprt_loop, DecayFit, FuzzerFn, SprtLoopResult, Verdict, K_MAX,
    K_MUT_MAX, K_MUT_MIN,
};

const PIPELINE_STAGES: usize = 3; // gate0, sprt, evaluator

/// What a single pipeline run concluded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeployVerdict {
    FullDeploymentGranted,
    DeploymentHalted,
}

impl DeployVerdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            DeployVerdict::FullDeploymentGranted => "FULL_DEPLOYMENT_GRANTED",
            DeployVerdict::DeploymentHalted => "DEPLOYMENT_HALTED",
        }
    }
}

/// Telemetry ingestion abstraction.  The concrete eBPF ringbuffer consumer can
/// implement this without changing the rest of the pipeline.
pub trait Ingestion {
    /// Drain one batch of telemetry streams (token sequences).
    fn next_batch(&mut self) -> Vec<Stream>;
}

/// Per-stage result collector.
pub struct PipelineReport {
    pub gate0: Option<Gate0Result>,
    pub sprt: Option<DecayFit>,
    /// Full output of the executed parallel fuzzer loop (distances, viability,
    /// entropy series) — populated by [`run_pipeline`].
    pub spr_loop: Option<SprtLoopResult>,
    pub evaluator: Option<EvaluatorResult>,
    pub verdict: DeployVerdict,
}

impl PipelineReport {
    pub fn stage_summary(&self) -> String {
        format!(
            "gate0={}, sprt={}, spr_rounds={}, evaluator={}",
            self.gate0.as_ref().map(|g| g.status.to_string()).unwrap_or_else(|| "n/a".into()),
            self.sprt.as_ref().map(|s| s.rd_verdict.as_str().to_string()).unwrap_or_else(|| "n/a".into()),
            self.spr_loop.as_ref().map(|s| s.k_used.to_string()).unwrap_or_else(|| "n/a".into()),
            self.evaluator.as_ref().map(|e| e.status.to_string()).unwrap_or_else(|| "n/a".into()),
        )
    }
}

/// `SystemSpec` bundles the data needed to run one pipeline evaluation.
pub struct SystemSpec {
    pub prod_streams: Vec<Stream>,
    pub cand_streams: Vec<Stream>,
    /// SPRT per-round distances (must be at least the 5-point window).
    pub sprt_distances: Vec<f64>,
    /// IPW inputs: prod_x, prod_y, prod_sel, eval_x, eval_y, eval_sel (f64 slices).
    pub ipw_prod_x: Vec<f64>,
    pub ipw_prod_y: Vec<f64>,
    pub ipw_prod_sel: Vec<f64>,
    pub ipw_eval_x: Vec<f64>,
    pub ipw_eval_y: Vec<f64>,
    pub ipw_eval_sel: Vec<f64>,
}

/// Run the deterministic pipeline over a system spec.
///
/// `sprt_fuzzer` is **actually executed** via [`run_sprt_loop`] (K in [3, 8], up
/// to `k_max` rounds); the resulting per-round distances are then fed to the
/// exponential-decay convergence diagnostic.  The precomputed
/// [`SystemSpec::sprt_distances`] window is used as an additional, independent
/// convergence check so a caller can supply either a live fuzzer or a fixed
/// recorded trajectory.
pub fn run_pipeline(spec: &SystemSpec, sprt_fuzzer: &FuzzerFn) -> PipelineReport {
    // Stage 1: Gate 0 distributional attestation.
    let g0 = gate0(&spec.prod_streams, &spec.cand_streams);

    // Stage 2: SPRT — run the parallel fuzzer loop, then fit the decay on the
    // fuzzer-produced distances (requires >= 5 rounds to fit).
    let loop_result: SprtLoopResult = run_sprt_loop(sprt_fuzzer, K_MAX, K_MUT_MIN, K_MUT_MAX, 0x5eed);
    let sprt_live = assess_convergence(&loop_result.distances);

    // Also assess the caller-supplied recorded window (independent check).
    let sprt_recorded = assess_convergence(&spec.sprt_distances);

    // The pipeline converges only if BOTH the live fuzzer fit and the recorded
    // window agree on a PASS verdict.
    let both_converged = sprt_live.converged && sprt_recorded.converged;
    let sprt = if both_converged {
        sprt_live
    } else if !sprt_live.converged {
        // Prefer the failing one so the report explains *why* it halted.
        sprt_live
    } else {
        sprt_recorded
    };

    // Stage 3: Phase-4 IPW correlation gate.
    let ev = evaluate_ipw(
        &spec.ipw_prod_x,
        &spec.ipw_prod_y,
        &spec.ipw_prod_sel,
        &spec.ipw_eval_x,
        &spec.ipw_eval_y,
        &spec.ipw_eval_sel,
    );

    let granted = g0.passed
        && both_converged
        && ev.passed
        && (sprt.rd_verdict == Verdict::Pass);
    let verdict = if granted {
        DeployVerdict::FullDeploymentGranted
    } else {
        DeployVerdict::DeploymentHalted
    };

    PipelineReport {
        gate0: Some(g0),
        sprt: Some(sprt),
        spr_loop: Some(loop_result),
        evaluator: Some(ev),
        verdict,
    }
}

/// A reference SPRT fuzzer whose per-round distance decays geometrically
/// (`d^(i) = 0.3^(i+1)`), producing a cleanly convergent trajectory for callers
/// that want a known-converging baseline without a real mutation engine.
pub fn reference_fuzzer() -> Box<FuzzerFn> {
    Box::new(|round: usize, _worker: usize| 0.3f64.powi((round + 1) as i32))
}

/// A fuzzer that never converges (jagged distances) — useful for negative
/// pipeline tests.
pub fn nonconverging_fuzzer() -> Box<FuzzerFn> {
    const PATTERN: [f64; 6] = [0.30, 0.10, 0.28, 0.05, 0.22, 0.30];
    Box::new(|round: usize, _worker: usize| PATTERN[round % PATTERN.len()])
}

pub const STAGE_COUNT: usize = PIPELINE_STAGES;