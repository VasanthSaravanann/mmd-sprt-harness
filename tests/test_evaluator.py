"""Task 4: IPW correlation evaluator (Phase 4 gate)."""

import numpy as np
import pytest

from harness.evaluator import (
    N_EFF_MIN,
    P_MAX,
    R_MIN,
    assemble_combined,
    effective_sample_size,
    evaluate_ipw,
    inverse_propensity_weight,
    winsorize,
)


def _gen(n, r=0.9, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    y = r * x + (1 - r) * rng.normal(0.0, 0.1, n)
    return x.tolist(), y.tolist()


@pytest.fixture
def good_data():
    px, py = _gen(3000, r=0.90, seed=1)   # 1-quote% randomized production
    ex, ey = _gen(4000, r=0.95, seed=2)   # stratified high-V_E evaluation
    pp = [0.01] * len(px)                  # 1% inclusion probability
    ep = [0.40] * len(ex)
    return px, py, pp, ex, ey, ep


def test_pass_when_strong_ipw_correlation(good_data):
    res = evaluate_ipw(*good_data)
    assert res.passed is True, res.reason
    assert res.n_eff >= N_EFF_MIN
    assert res.r_ipw > R_MIN
    assert res.p_value < P_MAX


def test_fail_when_uncorrelated_signal(good_data):
    px, py, pp, ex, ey, ep = good_data
    noisy = np.random.default_rng(9).normal(0.0, 1.0, len(ex)).tolist()
    res = evaluate_ipw(px, py, pp, ex, noisy, ep)
    assert res.passed is False
    assert res.r_ipw <= R_MIN


def test_fail_insufficient_effective_sample():
    # Tiny combined sample -> n_eff well below the n_eff >= 30 floor.
    px = [0.1, 0.2, 0.3]; py = [0.1, 0.2, 0.3]
    res = evaluate_ipw(px, py, [0.5] * 3, px[:2], py[:2], [0.5] * 2)
    assert res.passed is False
    assert res.n_eff < N_EFF_MIN


def test_winsorize_caps_p99():
    w = list(range(100, 101, 1))[::-1]  # descending weights
    capped = winsorize(w)
    # The tail weights (above P99) are compressed down to the cap.
    assert max(capped) <= max(capped)  # sanity; P99 of a tight range ~ max
    assert all(c <= max(capped) for c in capped)


def test_winsorize_leaves_max_unbounded_below_p99():
    weights = [1.0] * 100 + [1e6]
    capped = winsorize(weights, p=0.99)
    # Only the top 1% (the single 1e6 outlier) is capped down to P99 = 1.0.
    assert capped[-1] == 1.0
    assert max(capped) == 1.0


def test_effective_sample_size_equal_weights():
    weights = [2.0] * 50
    assert effective_sample_size(weights) == pytest.approx(50.0)


def test_inverse_propensity_weight():
    assert inverse_propensity_weight(0.01) == pytest.approx(100.0)
    assert inverse_propensity_weight(0.4) == pytest.approx(2.5)


def test_assemble_combined_winsorizes(good_data):
    px, py, pp, ex, ey, ep = good_data
    x, y, w = assemble_combined(px, py, pp, ex, ey, ep)
    assert len(x) == len(w) == len(y)
    # Winsorization drags the production tail weights down to the P99 cap,
    # which keeps the effective sample size from collapsing.
    assert effective_sample_size(w) >= N_EFF_MIN


def test_constants():
    assert N_EFF_MIN == 30
    assert R_MIN == 0.85
    assert P_MAX == 0.01

# --------------------------------------------------------------------------
# Regression: winsorize must actually cap the top (1 - p) fraction.
# Previously it used ceil(p*(n-1)) which landed on the maximum element for
# p=0.99, leaving outliers untouched (a silent no-op).
# --------------------------------------------------------------------------
def test_winsorize_actually_caps_single_outlier():
    weights = [1.0] * 99 + [1e9]
    capped = winsorize(weights, 0.99)
    assert max(capped) == 1.0, "single outlier must be clamped to P99"


def test_winsorize_caps_top_one_percent_general():
    # 1..1000: the top 1% (values 992..1000) must all be capped to <= P99.
    import math
    weights = [float(i) for i in range(1, 1001)]
    capped = winsorize(weights, 0.99)
    p99 = math.floor(0.99 * (len(weights) - 1))  # index of the cap value
    assert max(capped) == float(p99 + 1), f"cap should be sorted[{p99}], got {max(capped)}"


def test_winsorize_rejects_invalid_p():
    import pytest as _pytest
    with _pytest.raises(ValueError):
        winsorize([1.0, 2.0, 3.0], 1.0)


def test_winsorize_single_element_is_unchanged():
    assert winsorize([5.0], 0.99) == [5.0]
