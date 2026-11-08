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
    """
    w = list(weights)
    if not w:
        return w
    if not 0.0 < p < 1.0:
        raise ValueError(f"winsorization p must be in (0, 1), got {p}")
    n = len(w)
    if n == 1:
        return w
    sorted_w = sorted(w)
    idx = min(n - 2, int(math.floor(p * (n - 1))))
    cap = sorted_w[idx]
    return [min(x, cap) for x in w]


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
    wbar = wsum / n

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
    p = 2.0 * _survival_t(dof, abs(t_stat))
    return (r, p)


def _survival_t(dof: float, t: float) -> float:
    """Two-sided Student-t survival by numerical integration (normal approx)."""
    from math import exp, lgamma
    if dof > 120:
        # normal approximation
        return 0.5 * math.erfc(t / math.sqrt(2.0))
    steps = 4000
    xstart = t
    xend = 40.0
    if xstart >= xend:
        return 0.0
    dx = (xend - xstart) / steps
    s = 0.0
    half = lgamma((dof + 1) / 2) - lgamma(dof / 2) - 0.5 * math.log(dof * math.pi)
    for i in range(steps):
        xi = xstart + (i + 0.5) * dx
        s += dx * math.exp(half - (dof + 1) / 2 * math.log(1 + xi * xi / dof))
    return s


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
    passed: bool
    reason: str


def assemble_combined(
    prod_x: Sequence[float],
    prod_y: Sequence[float],
    prod_select_prob: Sequence[float],
    eval_x: Sequence[float],
    eval_y: Sequence[float],
    eval_select_prob: Sequence[float],
) -> tuple:
    """Combine 1% production + high-:math:`V_E` eval samples; winsorize IPW."""
    weights: List[float] = []
    x: List[float] = []
    y: List[float] = []

    # Keep a 1% randomized subset of production.
    prod_n = len(prod_x)
    keep = max(1, int(round(prod_n * PROD_FRACTION)))
    if keep > prod_n:
        keep = prod_n
    for i in range(keep):
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
) -> EvaluatorData:
    """Run the Phase 4 IPW correlation gate."""
    x, y, weights = assemble_combined(
        prod_x, prod_y, prod_select_prob,
        eval_x, eval_y, eval_select_prob,
    )
    n_eff = effective_sample_size(weights)
    # Use the weighted Spearman (rank) correlation to match the spec and the
    # Rust implementation; see weighted_spearman_corr for rationale.
    r_ipw, p_value = weighted_spearman_corr(x, y, weights)

    if n_eff < N_EFF_MIN:
        reason = f"n_eff={n_eff:.1f} < 30 (insufficient effective sample)"
        passed = False
    elif r_ipw <= R_MIN:
        reason = f"r_IPW={r_ipw:.3f} <= 0.85"
        passed = False
    elif p_value >= P_MAX:
        reason = f"p={p_value:.4f} >= 0.01 (not significant)"
        passed = False
    else:
        reason = f"n_eff={n_eff:.1f}, r_IPW={r_ipw:.3f} (p={p_value:.4f})"
        passed = True
    return EvaluatorData(x, y, weights, n_eff, r_ipw, p_value, passed, reason)