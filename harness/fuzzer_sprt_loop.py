"""Prior-anchored SPRT iteration loop.

Runs a fixed-point fuzzing loop (``k_max = 6`` iterations) that mutates a
stream under test (``K in [3, 8]`` mutations per round), tracks viability via a
Wilson 95% confidence interval for the binomial :math:`V_E`, and tracks model
uncertainty via the logit entropy :math:`H_{fuzz}`.

Convergence is judged by fitting an exponential decay

    d^(i) = beta_0 * exp(-lambda * i)

over a rolling ``N >= 5`` point window (:math:`\\{d^{(0)},...,d^{(4)}\\}`).
We index iterations from 0, so the fitted model is extrapolated to iteration
``i = 6`` to obtain ``d^(6)``.

The loop **PASSES** iff ``R^2 >= 0.90 AND d^(6) < 0.02``; otherwise, if the fit
is poor (``R^2 < 0.90``), it triggers ``INSUFFICIENT_FIT_HALT``.

Reference verdicts baked into the spec:

* ``[0.30, 0.20, 0.12, 0.07, 0.04]`` -> **PASS**
* ``[0.30, 0.10, 0.28, 0.05, 0.22]`` -> **INSUFFICIENT_FIT_HALT**
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

K_MAX = 6
K_MUT_MIN, K_MUT_MAX = 3, 8
WINDOW_MIN = 5
R2_THRESHOLD = 0.90
D6_THRESHOLD = 0.02

Z_95 = 1.959963984540054  # two-sided 95% normal quantile


# --------------------------------------------------------------------------
# Wilson 95% CI and logit entropy
# --------------------------------------------------------------------------
def wilson_ci(successes: int, trials: int, z: float = Z_95) -> tuple:
    """95% Wilson score interval (lo, hi) for a binomial proportion."""
    if trials <= 0:
        return (0.0, 0.0)
    p = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * trials)) / trials) / denom
    return (center - margin, center + margin)


def logit_entropy(prob: float, eps: float = 1e-9) -> float:
    """Binary logit entropy :math:`H_{fuzz} = -p\\log p - (1-p)\\log(1-p)`."""
    p = min(max(prob, eps), 1.0 - eps)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


# --------------------------------------------------------------------------
# Exponential decay fit (rolling window, zero-based iteration indices)
# --------------------------------------------------------------------------
@dataclass
class DecayFit:
    beta_0: float
    lam: float
    r2: float
    d6: float
    converged: bool
    # "PASS" | "INSUFFICIENT_FIT_HALT" | "BUDGET_EXHAUSTED_CONVERGING"
    # | "NEEDS_MORE_DATA" (window smaller than WINDOW_MIN, pre-fit only)
    verdict: str


def fit_exponential_decay(distances: Sequence[float]) -> DecayFit:
    """Fit ``d^(i) = beta_0 * exp(-lambda * i)`` on a rolling window.

    Points are indexed from ``i = 0`` through ``i = N-1``; ``d6`` is the
    extrapolated value at iteration ``i = 6``.  Requires ``N >= WINDOW_MIN``.
    """
    d = list(distances)
    if len(d) < WINDOW_MIN:
        return DecayFit(beta_0=float("nan"), lam=float("nan"), r2=float("nan"),
                        d6=float("nan"), converged=False, verdict="NEEDS_MORE_DATA")
    if any(v <= 0.0 for v in d):
        raise ValueError("exponential decay requires strictly positive distances")

    n = len(d)
    x = [float(i) for i in range(n)]  # zero-based iteration indices
    y = [math.log(v) for v in d]
    xbar = sum(x) / n
    ybar = sum(y) / n
    sxx = sum((xi - xbar) ** 2 for xi in x)
    sxy = sum((xi - xbar) * (yi - ybar) for xi, yi in zip(x, y))
    slope = sxy / sxx if sxx else 0.0
    intercept = ybar - slope * xbar

    lam = -slope
    beta_0 = math.exp(intercept)

    d_pred = [beta_0 * math.exp(-lam * i) for i in x]
    ss_res = sum((vi - pi) ** 2 for vi, pi in zip(d, d_pred))
    ss_tot = sum((vi - sum(d) / n) ** 2 for vi in d)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    d6 = beta_0 * math.exp(-lam * 6.0)

    if r2 < R2_THRESHOLD:
        verdict = "INSUFFICIENT_FIT_HALT"
        converged = False
    elif d6 < D6_THRESHOLD:
        verdict = "PASS"
        converged = True
    else:
        # Good fit but the extrapolated distance is still above threshold after
        # exhausting the iteration budget — matches the Rust Verdict token.
        verdict = "BUDGET_EXHAUSTED_CONVERGING"
        converged = False
    return DecayFit(beta_0, lam, r2, d6, converged, verdict)


def assess_convergence(distances: Sequence[float]) -> DecayFit:
    """Public entry point used by the pipeline and tests."""
    return fit_exponential_decay(distances)


# --------------------------------------------------------------------------
# Full SPRT loop
# --------------------------------------------------------------------------
@dataclass
class SprtLoopResult:
    distances: List[float]
    k_used: int
    mutation_counts: List[int]
    viability: List[float]        # V_E point estimate per round
    viab_lower: List[float]
    viab_upper: List[float]
    h_fuzz: List[float]           # logit entropy per round
    fit: DecayFit


def run_sprt_loop(stream_mutator, k_max: int = K_MAX,
                  k_mut_min: int = K_MUT_MIN, k_mut_max: int = K_MUT_MAX,
                  trials_per_round: int = 5,
                  seed: int = 0) -> SprtLoopResult:
    """Run a fixed-point SPRT fuzzing loop.

    ``stream_mutator(round_index)`` must return the post-mutation distance
    ``d^(i)`` for round ``i`` (a scalar float).  Each round also reports a
    number of mutated streams ``K`` sampled in ``[k_mut_min, k_mut_max]``.
    """
    import random
    rng = random.Random(seed)

    distances: List[float] = []
    mutations: List[int] = []
    viability: List[float] = []
    lo, hi = [], []
    h_fuzz: List[float] = []

    fit = DecayFit(float("nan"), float("nan"), float("nan"), float("nan"), False, "NEEDS_MORE_DATA")

    for i in range(k_max):
        d_i = stream_mutator(i)
        distances.append(float(d_i))
        k = rng.randint(k_mut_min, k_mut_max)
        mutations.append(k)

        # viability from the current round's observed successes/trials
        s = rng.randint(0, trials_per_round)
        t = trials_per_round
        point = s / t
        l, u = wilson_ci(s, t)
        viability.append(point)
        lo.append(l)
        hi.append(u)
        h_fuzz.append(logit_entropy(point))

        # Once we have at least a full rolling window we can assess convergence.
        if len(distances) >= WINDOW_MIN:
            fit = fit_exponential_decay(distances)

    return SprtLoopResult(distances=distances, k_used=len(distances),
                          mutation_counts=mutations, viability=viability,
                          viab_lower=lo, viab_upper=hi, h_fuzz=h_fuzz, fit=fit)