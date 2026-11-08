"""Phase 4 gate: IPW correlation evaluator.

Combines a 1% randomized production sample with a stratified, high-:math:`V_E`
evaluation sample.  Inverse-propensity weights are winsorized at the 99th
percentile to keep variance bounded and enforce ``n_eff >= 30``.

The gate **PASSES** iff ``n_eff >= 30`` **and** the IPW-weighted correlation
:math:`r_{2, IPW} > 0.85` with ``p < 0.01``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

N_EFF_MIN = 30
R_MIN = 0.85
P_MAX = 0.01
PROD_FRACTION = 0.01  # 1% randomized production sample


# --------------------------------------------------------------------------
# Weighting & winsorization
# --------------------------------------------------------------------------
def propensity_probability(select_prob: float) -> float:
    return max(1e-9, min(select_prob, 1.0))


def inverse_propensity_weight(select_prob: float) -> float:
    return 1.0 / propensity_probability(select_prob)


def winsorize(weights: Sequence[float], p: float = 0.99) -> List[float]:
    """Cap weights at the ``p``-th percentile (P99 by default).

    Uses the lower percentile rank ``floor(p * (n - 1))`` so that the top
    ``(1 - p)`` fraction of weights is genuinely clamped.  Using ``ceil`` would
    land on the maximum element for common ``p`` (e.g. ``p=0.99, n=100`` ->
    index 99) and silently leave outliers untouched.

    NaN-safe: NaN entries are excluded from the percentile rank calculation and
    preserved in the output, matching the Rust implementation exactly.
    """
    w = list(weights)
    if not w:
        return w
    if not 0.0 < p < 1.0:
        raise ValueError(f"winsorization p must be in (0, 1), got {p}")
    n = len(w)
    if n == 1:
        return w
    # NaN-safe sort: push NaN to end, non-NaN values sorted normally.
    # The key (v != v, v) sorts NaN (where v != v is True) after all finite values.
    sorted_w = sorted(w, key=lambda v: (v != v, v))
    # Exclude NaN from the rank calculation (matching Rust's non_nan filter).
    non_nan = sum(1 for v in sorted_w if v == v)
    if non_nan == 0:
        return w
    idx = min(non_nan - 2, int(math.floor(p * (non_nan - 1))))
    idx = max(idx, 0)
    cap = sorted_w[idx]
    # Preserve NaN entries: min(NaN, cap) in Python returns NaN (first arg
    # when neither comparison is True), matching Rust's explicit NaN guard.
    return [min(v, cap) for v in w]


def effective_sample_size(weights: Sequence[float]) -> float:
    w = list(weights)
    if not w:
        return 0.0
    total = sum(w)
    if total <= 0.0:
        return 0.0
    return total * total / sum(x * x for x in w)


def welch_t_approximation(n_eff: float) -> float:
    """Degrees-of-freedom proxy for the weighted sample."""
    return max(n_eff - 1.0, 1.0)


def _average_ranks(values: Sequence[float]) -> List[float]:
    """Average ranks with NaN-safe ordering (ties share the mean rank)."""
    order = sorted(range(len(values)), key=lambda i: (values[i] != values[i], values[i]))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        v = values[order[i]]
        while j + 1 < len(order) and values[order[j + 1]] == v:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in order[i : j + 1]:
            ranks[k] = avg
        i = j + 1
    return ranks


def weighted_spearman_corr(x: Sequence[float], y: Sequence[float],
                           weights: Sequence[float]) -> tuple:
    """Weighted **Spearman** rank correlation ``(r, p)``.

    Mirrors the Rust `weighted_spearman_corr`: rank both vectors, then compute a
    weighted Pearson correlation on the ranks.  Using Pearson on the raw values
    (as the old implementation did) diverges from the spec/Rust gate.
    """
    return weighted_correlation(_average_ranks(x), _average_ranks(y), weights)


def weighted_correlation(x: Sequence[float], y: Sequence[float],
                         weights: Sequence[float]) -> tuple:
    """IPW-weighted Pearson correlation ``(r, p)``."""
    n = len(x)
    if n < 3 or len(y) != n or len(weights) != n:
        raise ValueError("inputs must share length >= 3")
    w = list(weights)
    wsum = sum(w)
    if wsum <= 0:
        raise ValueError("non-positive weight sum")

    mx = sum(wi * xi for wi, xi in zip(w, x)) / wsum
    my = sum(wi * yi for wi, yi in zip(w, y)) / wsum

    cov = sum(wi * (xi - mx) * (yi - my) for wi, xi, yi in zip(w, x, y)) / wsum
    vx = sum(wi * (xi - mx) ** 2 for wi, xi in zip(w, x)) / wsum
    vy = sum(wi * (yi - my) ** 2 for wi, yi in zip(w, y)) / wsum

    denom = math.sqrt(vx * vy)
    if denom == 0.0:
        return (0.0, 1.0)
    r = cov / denom

    # Approximate t-test on n_eff degrees of freedom.
    n_eff = effective_sample_size(w)
    if abs(r) >= 1.0:
        return (r, 0.0)
    dof = welch_t_approximation(n_eff)
    t_stat = r * math.sqrt(dof / (1.0 - r * r))
    p = student_t_two_tail(abs(t_stat), dof)
    return (r, p)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction evaluation of the incomplete beta function."""
    MAXIT = 200
    EPS = 3.0e-16
    FPMIN = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    from math import lgamma
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        lgamma(a + b) - lgamma(a) - lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def student_t_two_tail(t: float, dof: float) -> float:
    """Exact two-sided Student-t p-value via the regularized incomplete beta.

    ``p = I_{dof/(dof + t^2)}(dof/2, 1/2)``.  This is the same algorithm used by
    the Rust implementation, so both languages return identical p-values (the
    previous Python version integrated the density numerically while Rust used a
    normal approximation, which could flip the ``p < 0.01`` gate across
    languages for small ``n_eff``).
    """
    if dof <= 0.0:
        return 1.0
    x = dof / (dof + t * t)
    return _betai(dof / 2.0, 0.5, x)


def _survival_t(dof: float, t: float) -> float:
    """One-sided Student-t survival ``P(T > t)`` (= half the two-sided p).

    .. deprecated::
        Production code uses :func:`student_t_two_tail` directly.  Kept for
        backward compatibility with ``tests/test_evaluator.py``.
    """
    return 0.5 * student_t_two_tail(t, dof)


# --------------------------------------------------------------------------
# Data assembly
# --------------------------------------------------------------------------
@dataclass
class EvaluatorData:
    x: List[float]
    y: List[float]
    weights: List[float]
    n_eff: float
    r_ipw: float
    p_value: float
    status: str
    passed: bool
    reason: str


def assemble_combined(
    prod_x: Sequence[float],
    prod_y: Sequence[float],
    prod_select_prob: Sequence[float],
    eval_x: Sequence[float],
    eval_y: Sequence[float],
    eval_select_prob: Sequence[float],
    seed: int = 0,
) -> tuple:
    """Combine 1% production + high-:math:`V_E` eval samples; winsorize IPW.

    The production subset is a *uniform random sample* of 1% of the rows
    (seeded for reproducibility).  The previous implementation took the first
    ``keep`` rows, which is a deterministic, order-biased subset and contradicts
    the spec's "1% randomized production data".
    """
    import random

    weights: List[float] = []
    x: List[float] = []
    y: List[float] = []

    # Keep a 1% randomized subset of production.
    prod_n = len(prod_x)
    # Use floor(x + 0.5) rather than Python's banker's-rounding round() so the
    # row count matches the Rust implementation exactly on .5 boundaries.
    keep = max(1, int(math.floor(prod_n * PROD_FRACTION + 0.5)))
    keep = min(keep, prod_n)
    rng = random.Random(seed)
    idxs = rng.sample(range(prod_n), keep)
    for i in idxs:
        x.append(prod_x[i])
        y.append(prod_y[i])
        weights.append(inverse_propensity_weight(prod_select_prob[i]))

    # Entire stratified high-V_E evaluation sample.
    for i in range(len(eval_x)):
        x.append(eval_x[i])
        y.append(eval_y[i])
        weights.append(inverse_propensity_weight(eval_select_prob[i]))

    weights = winsorize(weights)
    return x, y, weights


def evaluate_ipw(
    prod_x: Sequence[float],
    prod_y: Sequence[float],
    prod_select_prob: Sequence[float],
    eval_x: Sequence[float],
    eval_y: Sequence[float],
    eval_select_prob: Sequence[float],
    seed: int = 0,
) -> EvaluatorData:
    """Run the Phase 4 IPW correlation gate.

    ``seed`` controls the random 1% production subsample (see
    :func:`assemble_combined`).
    """
    x, y, weights = assemble_combined(
        prod_x, prod_y, prod_select_prob,
        eval_x, eval_y, eval_select_prob,
        seed=seed,
    )
    n_eff = effective_sample_size(weights)
    # Use the weighted Spearman (rank) correlation to match the spec and the
    # Rust implementation; see weighted_spearman_corr for rationale.
    r_ipw, p_value = weighted_spearman_corr(x, y, weights)

    if n_eff < N_EFF_MIN:
        status = "INSUFFICIENT_POWER"
        reason = f"n_eff={n_eff:.1f} < 30 (insufficient effective sample)"
        passed = False
    elif r_ipw <= R_MIN:
        status = "INSUFFICIENT_CORRELATION"
        reason = f"r_IPW={r_ipw:.3f} <= 0.85"
        passed = False
    elif p_value >= P_MAX:
        status = "NOT_SIGNIFICANT"
        reason = f"p={p_value:.4f} >= 0.01 (not significant)"
        passed = False
    else:
        status = "PASS"
        reason = f"n_eff={n_eff:.1f}, r_IPW={r_ipw:.3f} (p={p_value:.4f})"
        passed = True
    return EvaluatorData(x, y, weights, n_eff, r_ipw, p_value, status, passed, reason)