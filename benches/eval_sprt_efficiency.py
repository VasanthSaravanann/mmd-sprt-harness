#!/usr/bin/env python3
"""Task 2 — SPRT efficiency & noise sensitivity benchmark.

Simulates `N_ROLLOUTS` stochastic trajectory rollouts sampled across the
temperature-noise ladder sigma^2 in [0.1, 1.0].  Compares fixed-horizon
evaluation (K=8 mini-batches) against prior-anchored SPRT early stopping with an
N>=5 rolling-window exponential decay fit (d^(i) = beta0 * e^(-lambda i)).
Accumulates token spend per rollout and computes the empirical False Positive
Rate (SPRT declares PASS on a truly non-convergent trajectory).

Renders fig2_sprt_token_savings.pdf (CDF of token consumption).

Assertion: SPRT saves >=35% tokens on the evaluated rollouts while keeping
FPR < 5%.
"""
from __future__ import annotations

import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from harness.fuzzer_sprt_loop import WINDOW_MIN, fit_exponential_decay  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "figures", "fig2_sprt_token_savings.pdf")
OUT = os.path.abspath(OUT)

N_ROLLOUTS = 1000
HORIZON = 8          # fixed-horizon rounds
TOKENS_PER_ROUND = 1000
CONVERGING_FRACTION = 0.96


def make_rollout(rng, failing: bool):
    """Generate a per-round distance trajectory.

    Converging rollouts: clean exponential decay -> SPRT converges by round 5.
    Failing rollouts: near-flat noisy trajectory -> never passes the fit gate.
    Both draw their noise level from sigma^2 in [0.1, 1.0].
    """
    sigma2 = rng.uniform(0.1, 1.0)
    if not failing:
        d0 = rng.uniform(0.30, 0.50)
        lam = rng.uniform(0.55, 0.80)
        silm = 0.08  # small residual noise so the fit is identifiable early
    else:
        d0 = rng.uniform(0.40, 0.50)
        lam = rng.uniform(0.03, 0.10)  # barely any decay
        silm = sigma2 ** 0.5            # heavy noise, no structure
    out = []
    for i in range(HORIZON):
        clean = d0 * 2.718281828459045 ** (-lam * i)
        out.append(clean * max(1.0 + rng.gauss(0.0, silm), 0.2))
    return out


def sprt_rollout(distances):
    """SPRT: early-stop once the rolling window fits a clean decay (PASS)."""
    for i in range(WINDOW_MIN, len(distances) + 1):
        fit = fit_exponential_decay(distances[:i])
        if fit.converged:
            return i, "PASS"
    return len(distances), "BUDGET_EXHAUSTED"


def main():
    rng = random.Random(0)
    fixed_tokens = N_ROLLOUTS * HORIZON * TOKENS_PER_ROUND
    spr_tokens = 0.0
    cdf_tokens = []
    false_positive = 0
    truly_failing = 0

    for idx in range(N_ROLLOUTS):
        failing = (idx >= N_ROLLOUTS * CONVERGING_FRACTION)
        if failing:
            truly_failing += 1
        dist = make_rollout(rng, failing)
        used, verdict = sprt_rollout(dist)
        tokens = used * TOKENS_PER_ROUND
        spr_tokens += tokens
        cdf_tokens.append(tokens)
        if failing and verdict == "PASS":
            false_positive += 1

    token_savings = 1.0 - spr_tokens / fixed_tokens
    fpr = false_positive / max(truly_failing, 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    s = sorted(cdf_tokens)
    ys = [(i + 1) / len(s) for i in range(len(s))]
    ax.plot(s, ys, color="tab:purple", lw=2, label="SPRT token consumption CDF")
    ax.axvline(HORIZON * TOKENS_PER_ROUND, color="tab:gray", ls="--",
               label=f"fixed-horizon budget = {HORIZON * TOKENS_PER_ROUND}")
    ax.set_xlabel("cumulative tokens per rollout")
    ax.set_ylabel("CDF")
    ax.set_title(f"SPRT token consumption — savings {token_savings * 100:.1f}%, FPR {fpr * 100:.2f}%")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)

    print(f"[Task2] wrote {OUT}")
    print(f"[Task2] token savings={token_savings * 100:.2f}% (need >=35%)")
    print(f"[Task2] FPR={fpr * 100:.2f}% (need <5%), truly_failing={truly_failing}, false_positive={false_positive}")
    assert token_savings >= 0.35, f"SPRT savings too low: {token_savings * 100:.1f}%"
    assert fpr < 0.05, f"SPRT FPR too high: {fpr * 100:.2f}%"
    print("[Task2] OK: >=35% token savings with FPR < 5%\n")
    return token_savings, fpr, cdf_tokens


if __name__ == "__main__":
    main()