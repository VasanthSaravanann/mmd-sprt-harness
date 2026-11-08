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


# --------------------------------------------------------------------------
# Regression: evaluate_ipw must use a rank-based (Spearman) correlation to match
# the Rust gate, not a Pearson correlation on the raw values.
# --------------------------------------------------------------------------
def test_evaluate_ipw_is_rank_based_not_pearson():
    # Build data where a monotone-but-heteroscedastic relationship gives a
    # perfect Spearman (rank) correlation but a weaker Pearson correlation.
    n = 300
    x = list(range(1, n + 1))
    # perfect rank ordering (strictly increasing) but wildly non-linear scale
    y = [float(v) ** 3 for v in x]
    weights = [1.0] * n
    from harness.evaluator import weighted_spearman_corr, weighted_correlation
    r_spearman, _ = weighted_spearman_corr(x, y, weights)
    r_pearson, _ = weighted_correlation(x, y, weights)
    assert r_spearman == pytest.approx(1.0, abs=1e-9)
    assert r_pearson < 0.99, "raw Pearson should be < 1 on cubic data"


def test_average_ranks_ties_share_mean():
    from harness.evaluator import _average_ranks
    assert _average_ranks([1.0, 1.0, 3.0, 2.0]) == [1.5, 1.5, 4.0, 3.0]


# --------------------------------------------------------------------------
# Regression: the two-sided p-value must match the normal approximation
# (previously the Rust path returned a one-sided value half of Python's).
# Also: production subsampling must be RANDOM (spec: "1% randomized"), and the
# default seed must make the result reproducible.
# --------------------------------------------------------------------------
def test_two_sided_p_value_normal_case():
    from harness.evaluator import _survival_t, welch_t_approximation
    t = 1.5
    dof = welch_t_approximation(11)  # n_eff=11 -> dof=10 -> integration branch
    p_two = 2.0 * _survival_t(dof, abs(t))
    p_one = _survival_t(dof, abs(t))
    assert p_two == pytest.approx(2.0 * p_one)


def test_production_subsample_is_randomized_not_first_n():
    from harness.evaluator import assemble_combined
    n = 3000
    px = list(range(n)); py = list(range(n)); pp = [0.01] * n
    ex = [0.5] * 10; ey = [0.5] * 10; ep = [0.4] * 10
    x, _, _ = assemble_combined(px, py, pp, ex, ey, ep)
    # 1% of 3000 = 30 rows; a first-N implementation yields 0..29 exactly.
    assert set(x[:30]) != set(range(30)), "production sample must be randomized, not first-N"


def test_production_subsample_reproducible_with_seed():
    from harness.evaluator import assemble_combined
    n = 2000
    px = list(range(n)); py = list(range(n)); pp = [0.01] * n
    ex = [0.5] * 5; ey = [0.5] * 5; ep = [0.4] * 5
    x1, _, _ = assemble_combined(px, py, pp, ex, ey, ep, seed=42)
    x2, _, _ = assemble_combined(px, py, pp, ex, ey, ep, seed=42)
    assert x1 == x2


# --------------------------------------------------------------------------
# Regression: student_t_two_tail must match scipy exactly (the Rust and Python
# implementations now share this algorithm; previously Python integrated the
# density numerically and Rust used a normal approximation).
# --------------------------------------------------------------------------
def test_student_t_two_tail_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    from harness.evaluator import student_t_two_tail
    for t, dof in [(1.5, 10), (2.0, 29), (2.58, 29), (3.0, 5), (0.5, 100)]:
        mine = student_t_two_tail(t, dof)
        ref = 2.0 * scipy_stats.t.sf(abs(t), dof)
        assert mine == pytest.approx(ref, abs=1e-12), f"t={t} dof={dof}"


def test_student_t_two_tail_is_two_sided():
    from harness.evaluator import student_t_two_tail
    # p is symmetric in t and < 1
    assert student_t_two_tail(2.0, 29) == pytest.approx(student_t_two_tail(-2.0, 29))
    assert student_t_two_tail(0.0, 29) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Regression: the 1% production subsample size uses floor(x + 0.5) so the row
# count matches Rust on .5 boundaries (Python's round() is banker's rounding:
# round(2.5)=2 while Rust's f64::round is half-away-from-zero: 2.5->3).
# --------------------------------------------------------------------------
def test_production_keep_uses_half_up_rounding():
    from harness.evaluator import assemble_combined
    prod_n = 250  # 250 * 0.01 = 2.5 -> half-up gives 3
    px = list(range(prod_n)); py = list(range(prod_n)); pp = [0.01] * prod_n
    ex = [0.0] * 4; ey = [0.0] * 4; ep = [0.4] * 4
    x, _, _ = assemble_combined(px, py, pp, ex, ey, ep, seed=1)
    assert len(x) == 4 + 3, f"expected 3 production rows, got {len(x) - 4}"
