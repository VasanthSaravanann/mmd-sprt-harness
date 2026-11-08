"""Task 5: master pipeline integration test (Gate 0 -> Phase 4)."""

import random

import numpy as np

from harness.gate0 import gate0
from harness.pipeline import (
    STAGES,
    SystemUnderTest,
    run_pipeline,
    stage_summary,
)

MOTIFS = [["a", "b", "b", "a"], ["a", "a", "b", "a"], ["b", "a", "a", "b"]]


def _motif_stream(motif, length=60):
    return [motif[i % len(motif)] for i in range(length)]


def _make_prod(seed=123):
    rng = random.Random(seed)
    return [_motif_stream(rng.choice(MOTIFS)) for _ in range(90)]


def _healthy_cand(prod, seed=124):
    rng = random.Random(seed)
    out = []
    for s in prod:
        s = list(s)
        for _ in range(int(round(len(s) * 0.002))):
            s[rng.randrange(len(s))] = rng.choice(["a", "b"])
        out.append(s)
    return out


def _exfil_cand(prod, seed=125):
    rng = random.Random(seed)
    out = []
    for s in prod:
        s = list(s)
        k = max(1, int(round(len(s) * 0.01)))
        for _ in range(k):
            s[rng.randrange(len(s))] = "S1"
        out.append(s)
    return out


def _gen(n, r=0.92, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    y = r * x + (1 - r) * rng.normal(0.0, 0.1, n)
    return x.tolist(), y.tolist()


def _good_ipw():
    px, py = _gen(3000, r=0.9, seed=1)
    ex, ey = _gen(4000, r=0.95, seed=2)
    return px, py, [0.01] * len(px), ex, ey, [0.4] * len(ex)


def _bad_ipw():
    px, py, pp, ex, ey, ep = _good_ipw()
    ey = np.random.default_rng(9).normal(0.0, 1.0, len(ex)).tolist()
    return px, py, pp, ex, ey, ep


def _valid_telemetry(n=4):
    return [
        {"ts": float(i), "event_id": f"e{i}", "tier": 2,
         "token": "tk", "logit": 0.4, "kind": "near_miss"}
        for i in range(n)
    ]


def _invalid_telemetry():
    return [
        {"event_id": "e0", "tier": 2, "token": "tk", "logit": 0.4, "kind": "near_miss"},  # no ts
        {"ts": 0.0, "event_id": "e1", "tier": 99, "token": "tk", "logit": 0.4, "kind": "near_miss"},
    ]


def _robust_system():
    prod = _make_prod()
    return SystemUnderTest(
        projection=(prod, _healthy_cand(prod)),
        sprt_distances=[0.30, 0.20, 0.12, 0.07, 0.04],
        ipw_data=_good_ipw(),
        telemetry=_valid_telemetry(),
    )


def _flawed_system():
    prod = _make_prod()
    return SystemUnderTest(
        projection=(prod, _exfil_cand(prod)),
        sprt_distances=[0.30, 0.10, 0.28, 0.05, 0.22],
        ipw_data=_bad_ipw(),
        telemetry=_invalid_telemetry(),
    )


def test_robust_system_passes_full_pipeline():
    report = run_pipeline(_robust_system())
    assert report.overall is True, stage_summary(report)
    for s in report.stages:
        assert s.passed is True, f"{s.name}: {s.detail}"


def test_flawed_baseline_fails_full_pipeline():
    report = run_pipeline(_flawed_system())
    assert report.overall is False, stage_summary(report)
    # Every stage is expected to fail for the flawed baseline.
    for s in report.stages:
        assert s.passed is False, f"{s.name} unexpectedly passed: {s.detail}"


def test_pipeline_rejects_each_individual_weakness():
    prod = _make_prod()
    base = _robust_system()

    m = replace_with(base, "projection", (prod, _exfil_cand(prod)))
    report_mmd = run_pipeline(m)
    assert report_mmd.overall is False
    assert report_mmd.stage("gate0").passed is False

    # spr
    m_sprt = replace_with(base, "sprt_distances", [0.30, 0.10, 0.28, 0.05, 0.22])
    r_sprt = run_pipeline(m_sprt)
    assert r_sprt.stage("sprt").passed is False
    assert r_sprt.overall is False

    m_ipw = replace_with(base, "ipw_data", _bad_ipw())
    r_ipw = run_pipeline(m_ipw)
    assert r_ipw.stage("evaluator").passed is False
    assert r_ipw.overall is False

    m_ver = replace_with(base, "telemetry", _invalid_telemetry())
    r_ver = run_pipeline(m_ver)
    assert r_ver.stage("verifier").passed is False
    assert r_ver.overall is False


def replace_with(sys_: SystemUnderTest, attr: str, value):
    import dataclasses
    return dataclasses.replace(sys_, **{attr: value})


def test_all_stages_present():
    report = run_pipeline(_robust_system())
    assert [s.name for s in report.stages] == list(STAGES)


def test_gate0_alone_flips_between_systems():
    r1 = gate0(*_robust_system().projection)
    r2 = gate0(*_flawed_system().projection)
    assert r1.passed is True
    assert r2.passed is False

# --------------------------------------------------------------------------
# Regression: the verifier stage must actually route SIGKILL (previously
# gate.route() was never called, so the latency check was vacuously True).
# --------------------------------------------------------------------------
def test_verifier_stage_routes_sigkill_latency():
    # Tier-1 event with no pending Tier-2 triggers route_sigkill; with a
    # simulated latency just over budget the verifier stage must FAIL.
    tele = [{"ts": 0.0, "event_id": "e1", "tier": 1, "token": "tk",
             "logit": 0.4, "kind": "near_miss"}]
    sut = SystemUnderTest(
        projection=_robust_system().projection,
        sprt_distances=[0.30, 0.20, 0.12, 0.07, 0.04],
        ipw_data=_good_ipw(),
        telemetry=tele,
    )
    report = run_pipeline(sut, sigkill_latency_us=2500.0)  # 2.5ms > 2ms budget
    assert report.stage("verifier").passed is False
    report_ok = run_pipeline(sut, sigkill_latency_us=125.0)
    assert report_ok.stage("verifier").passed is True
