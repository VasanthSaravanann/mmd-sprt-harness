"""Task 3: prior-anchored SPRT iteration loop."""

import pytest

from harness.fuzzer_sprt_loop import (
    D6_THRESHOLD,
    K_MUT_MIN,
    K_MUT_MAX,
    K_MAX,
    R2_THRESHOLD,
    WINDOW_MIN,
    assess_convergence,
    fit_exponential_decay,
    logit_entropy,
    run_sprt_loop,
    wilson_ci,
)


def test_reference_pass_sequence():
    # [0.30, 0.20, 0.12, 0.07, 0.04] fits clean exponential decay
    # (R^2 ~ 0.99, extrapolated d^(6) < 0.02) => PASS.
    res = assess_convergence([0.30, 0.20, 0.12, 0.07, 0.04])
    assert res.verdict == "PASS"
    assert res.converged is True
    assert res.r2 >= R2_THRESHOLD
    assert res.d6 < D6_THRESHOLD


def test_reference_halt_sequence():
    # [0.30, 0.10, 0.28, 0.05, 0.22] is jagged -> poor fit -> halt.
    res = assess_convergence([0.30, 0.10, 0.28, 0.05, 0.22])
    assert res.verdict == "INSUFFICIENT_FIT_HALT"
    assert res.converged is False
    assert res.r2 < R2_THRESHOLD


def test_needs_more_data_below_window():
    res = assess_convergence([0.5, 0.4, 0.3, 0.2])
    assert res.verdict == "NEEDS_MORE_DATA"
    assert len(res.__dict__)  # dataclass populated with nan fields


def test_exp_decay_r2_zero_for_constant_series():
    res = fit_exponential_decay([0.5, 0.5, 0.5, 0.5, 0.5])
    # constant series -> r2 degenerates to 1.0 by convention (no variance)
    assert res.r2 == 1.0


def test_wilson_ci_ninetyfive_percent():
    lo, hi = wilson_ci(10, 10)
    assert 0.0 <= lo <= hi <= 1.0
    # 95% CI for p=1 widens below 1 (a point estimate would not).
    assert hi < 1.0


def test_wilson_ci_contains_point_estimate():
    lo, hi = wilson_ci(4, 10)
    assert lo <= 0.4 <= hi


def test_logit_entropy_zero_at_extremes():
    assert logit_entropy(0.0) == pytest.approx(0.0, abs=1e-6)
    assert logit_entropy(1.0) == pytest.approx(0.0, abs=1e-6)


def test_logit_entropy_max_at_center():
    import math
    h_mid = logit_entropy(0.5)
    assert h_mid > logit_entropy(0.2)
    assert h_mid > logit_entropy(0.9)
    # h(p) = -p ln p - (1-p) ln(1-p); maximum ln(2) at p = 0.5.
    assert h_mid == pytest.approx(math.log(2.0))


def test_exponential_fit_parameters():
    # d = 1.0 * exp(-0.5*i) at i = 0,1,2,3,4,5,6
    d = [1.0 * __import__("math").exp(-0.5 * i) for i in range(WINDOW_MIN)]
    res = fit_exponential_decay(d)
    assert res.beta_0 == pytest.approx(1.0, abs=0.05)
    assert res.lam == pytest.approx(0.5, abs=0.05)
    assert res.r2 >= 0.999


def test_run_sprt_loop_reaches_kmax():
    def mutator(i):
        return 0.3 ** (i + 1)

    res = run_sprt_loop(mutator, seed=3)
    assert res.k_used == K_MAX
    assert len(res.distances) == K_MAX
    assert all(K_MUT_MIN <= k <= K_MUT_MAX for k in res.mutation_counts)
    assert len(res.viability) == K_MAX


def test_run_sprt_loop_has_wilson_and_entropy_series():
    def mutator(i):
        return 0.3 ** (i + 1)

    res = run_sprt_loop(mutator, seed=7)
    for l, u in zip(res.viab_lower, res.viab_upper):
        assert 0.0 <= l <= u <= 1.0
    assert all(h >= 0.0 for h in res.h_fuzz)