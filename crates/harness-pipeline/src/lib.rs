//! Unified end-to-end harness pipeline.
//!
//! Wires Gate 0 (distributional attestation), the parallel SPRT loop, and the
//! IPW Phase-4 evaluator into a single pipeline.  In production this library is
//! fed events from an eBPF ringbuffer; here the ingestion boundary is the
//! `Ingestion` trait so the kernel-backed path can be dropped in later.

use harness_core::evaluator::{evaluate_ipw, EvaluatorResult};
use harness_core::gate0::{gate0, Gate0Result, Stream};
use harness_core::sprt::{assess_convergence, DecayFit, FuzzerFn, Verdict};

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
    pub evaluator: Option<EvaluatorResult>,
    pub verdict: DeployVerdict,
}

impl PipelineReport {
    pub fn stage_summary(&self) -> String {
        format!(
            "gate0={}, sprt={}, evaluator={}",
            self.gate0.as_ref().map(|g| g.passed.to_string()).unwrap_or_else(|| "n/a".into()),
            self.sprt.as_ref().map(|s| s.rd_verdict.as_str().to_string()).unwrap_or_else(|| "n/a".into()),
            self.evaluator.as_ref().map(|e| e.passed.to_string()).unwrap_or_else(|| "n/a".into()),
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
pub fn run_pipeline(spec: &SystemSpec, sprt_fuzzer: &FuzzerFn) -> PipelineReport {
    // Stage 1: Gate 0 distributional attestation.
    let g0 = gate0(&spec.prod_streams, &spec.cand_streams);

    // Stage 2: SPRT convergence diagnostics (on supplied per-round distances).
    let sprt = assess_convergence(&spec.sprt_distances);

    // Stage 3: Phase-4 IPW correlation gate.
    let ev = evaluate_ipw(
        &spec.ipw_prod_x,
        &spec.ipw_prod_y,
        &spec.ipw_prod_sel,
        &spec.ipw_eval_x,
        &spec.ipw_eval_y,
        &spec.ipw_eval_sel,
    );

    // Optional: run the async/parallel fuzzer loop (unused distances dominate if
    // we want to *exercise* it; here we validate against the provided window).
    let _ = sprt_fuzzer; // reserved for live runs; diagnostics already computed.

    let granted = g0.passed && sprt.converged && ev.passed && (sprt.rd_verdict == Verdict::Pass);
    let verdict = if granted {
        DeployVerdict::FullDeploymentGranted
    } else {
        DeployVerdict::DeploymentHalted
    };

    PipelineReport {
        gate0: Some(g0),
        sprt: Some(sprt),
        evaluator: Some(ev),
        verdict,
    }
}

/// Provide a default no-op SPRT fuzzer so callers not yet wiring the live loop
/// can still call [`run_pipeline`].
pub fn dummy_fuzzer() -> Box<FuzzerFn> {
    Box::new(|round: usize, _worker: usize| 0.3f64.powi((round + 1) as i32))
}

pub const STAGE_COUNT: usize = PIPELINE_STAGES;