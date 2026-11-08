"""Master pipeline: connects Gate 0 -> SPRT -> Phase 4 IPW evaluator -> verifier.

Each stage is an independent gate; the master pipeline emits a single PASS/FAIL
after every stage has reported.  A system is declared robust only when all four
stages pass; otherwise it is rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harness.evaluator import evaluate_ipw
from harness.fuzzer_sprt_loop import assess_convergence
from harness.gate0 import gate0
from harness.mock_verifier import (
    CascadeRouter,
    MockSigkillGate,
    sub_budget,
    validate_telemetry,
)

STAGES = ("gate0", "sprt", "evaluator", "verifier")


@dataclass
class SystemUnderTest:
    projection: tuple  # (prod_streams, cand_streams) for Gate 0
    sprt_distances: list
    ipw_data: tuple  # (prod_x, prod_y, prod_prop, eval_x, eval_y, eval_prop)
    telemetry: list   # list of telemetry records to run through the verifier


@dataclass
class StageReport:
    name: str
    passed: bool
    detail: str = ""
    extra: object = None


@dataclass
class PipelineReport:
    overall: bool
    stages: list = field(default_factory=list)

    def stage(self, name: str) -> StageReport:
        for s in self.stages:
            if s.name == name:
                return s
        return StageReport(name, False, "not executed")


def run_pipeline(system: SystemUnderTest,
                 sigkill_latency_us: float = 125.0) -> PipelineReport:
    stages: list[StageReport] = []

    # 1. Gate 0 distributional attestation.
    prod, cand = system.projection
    g0 = gate0(prod, cand)
    # Detail carries the canonical status token (uniform with the other stages).
    stages.append(StageReport("gate0", g0.passed, g0.status, g0))

    # 2. Prior-anchored SPRT convergence.
    sprt = assess_convergence(system.sprt_distances)
    stages.append(StageReport("sprt", sprt.converged, sprt.verdict, sprt))

    # 3. Phase 4 IPW correlation gate.
    px, py, pp, ex, ey, ep = system.ipw_data
    ipw = evaluate_ipw(px, py, pp, ex, ey, ep)
    # Detail carries the canonical status token so every stage reports a
    # uniform verdict string (Gate 0 reason, SPRT verdict, Phase-4 status).
    stages.append(StageReport("evaluator", ipw.passed, ipw.status, ipw))

    # 4. Userspace verifier: schema validity + SIGKILL routing budget.
    router = CascadeRouter()
    gate = MockSigkillGate(latency_us=sigkill_latency_us)
    schema_ok = True
    schema_errors = ""
    for record in system.telemetry:
        errs = validate_telemetry(record)
        if errs:
            schema_ok = False
            schema_errors = "; ".join(errs)
            break
        action, _ = router.ingest(record)
        # Previously gate.route() was never invoked here, so `gate.calls` stayed
        # empty and the latency check was vacuously True.  Actually route the
        # SIGKILL whenever the router escalates an event, mirroring
        # mock_verifier._VerifierHandler.
        if action == "route_sigkill":
            gate.route()
    latency_ok = sub_budget(gate.max_latency_us) if gate.calls else True
    verifier_ok = schema_ok and latency_ok
    detail = "schema;SIGKILL<2ms" if verifier_ok else (
        f"schema_ok={schema_ok} latency_ok={latency_ok} {schema_errors}".rstrip()
    )
    stages.append(StageReport("verifier", verifier_ok, detail))

    overall = all(s.passed for s in stages)
    return PipelineReport(overall=overall, stages=stages)


def stage_summary(report: PipelineReport) -> str:
    return ", ".join(f"{s.name}={s.passed}" for s in report.stages)