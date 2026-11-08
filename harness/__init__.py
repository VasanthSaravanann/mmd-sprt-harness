"""Distributional attestation + SPRT verification harness (abstraction proof)."""

from harness.gate0 import gate0, Gate0Result, MMD2_FLOOR, D_ACT_THRESHOLD, SIGMA_BW
from harness.evaluator import evaluate_ipw, EvaluatorData, N_EFF_MIN, R_MIN, P_MAX
from harness.fuzzer_sprt_loop import (
    assess_convergence, fit_exponential_decay, run_sprt_loop,
    DecayFit, SprtLoopResult, K_MAX, K_MUT_MIN, K_MUT_MAX,
    WINDOW_MIN, R2_THRESHOLD, D6_THRESHOLD,
)
from harness.mock_verifier import (
    CascadeRouter, MockSigkillGate, MockVerifierServer,
    validate_telemetry, sub_budget,
)
from harness.pipeline import run_pipeline, PipelineReport, SystemUnderTest, StageReport, STAGES

__all__ = [
    "gate0", "Gate0Result", "MMD2_FLOOR", "D_ACT_THRESHOLD", "SIGMA_BW",
    "evaluate_ipw", "EvaluatorData", "N_EFF_MIN", "R_MIN", "P_MAX",
    "assess_convergence", "fit_exponential_decay", "run_sprt_loop",
    "DecayFit", "SprtLoopResult", "K_MAX", "K_MUT_MIN", "K_MUT_MAX",
    "WINDOW_MIN", "R2_THRESHOLD", "D6_THRESHOLD",
    "CascadeRouter", "MockSigkillGate", "MockVerifierServer",
    "validate_telemetry", "sub_budget",
    "run_pipeline", "PipelineReport", "SystemUnderTest", "StageReport", "STAGES",
]