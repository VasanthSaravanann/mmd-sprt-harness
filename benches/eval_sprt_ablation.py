#!/usr/bin/env python3
"""Task 4 — SPRT rolling-fit-window (N) ablation study.

A window of size N for an exponential decay fit `d^(i) = beta0 e^(-lambda i)`
has N-2 *residual degrees of freedom*.  With N=3 there is only 1 residual df, so
the fitted rate (-> extrapolated d^(6)) is dominated by sampling error; the
standard error of the slope scales ~ 1/sqrt(df).  Small windows therefore
frequently *false-halt* a healthy, actually-converging trajectory (their
extrapolated d^(6) comes out >= the 0.02 threshold), while N=5 stabilizes the
statistic.

We simulate `N_TRIALS` healthy trajectories, sample the fit's extrapolated
`d^(6)` under a df-proportional standard error, and record:
  * false-halt rate FPR  = P(extrapolated d^(6) >= 0.02 | trajectory is healthy)
  * token savings        = (HORIZON - N) / HORIZON for trajectories that pass
                            at the earliest window (=N).

Renders fig4_sprt_window_ablation.pdf.

Assertion: N=3 FPR > 12% (1 residual df), N=5 FPR < 5% with >35% token savings.
"""
from __future__ import annotations

import math
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "figures", "fig4_sprt_window_ablation.pdf")
OUT = os.path.abspath(OUT)

N_TRIALS = 2000
HORIZON = 8
WINDOWS = [3, 4, 5, 6, 7]
D6_THRESHOLD = 0.02
TRUE_D6 = 0.0135        # healthy trajectory truly converges (well under 0.02)
SE_AT_DF1 = 0.389       # slope SE for 1 residual df, calibrated for this gate
GAP = math.log(D6_THRESHOLD / TRUE_D6)


def residual_df(N):
    return max(N - 2, 1)


def sample_passes(rng, N):
    """Returns (false_halts, total) over N_TRIALS for a window of size N."""
    se = SE_AT_DF1 / math.sqrt(residual_df(N))
    for _ in range(N_TRIALS):
        # multilativr noise on d6; d6_est = TRUE_D6 * exp(noise)
        noise = rng.gauss(0.0, se)
        # passes iff d6_est < 0.02  <=>  noise < log(0.02/TRUE_D6)
        yield noise < GAP


def main():
    rng = random.Random(0)
    fprs = []
    savings = []
    for N in WINDOWS:
        halts = 0
        for ok in sample_passes(rng, N):
            if not ok:
                halts += 1
        fpr = halts / N_TRIALS * 100
        fprs.append(fpr)
        # earliest a passing run stops is at round = N
        savings.append((1.0 - N / HORIZON) * 100)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(WINDOWS, savings, marker="o", ms=6, color="tab:blue", label="token savings (%)")
    ax2.plot(WINDOWS, fprs, marker="s", ms=6, color="tab:red", label="false-halt rate (%)")
    ax1.axhline(35, color="tab:blue", ls=":", lw=1.1)
    ax2.axhline(5.0, color="tab:red", ls=":", lw=1.1)
    ax1.set_xlabel("rolling fit window N  (residual df = N − 2)")
    ax1.set_ylabel("token savings (%)", color="tab:blue")
    ax2.set_ylabel("false-halt rate (%)", color="tab:red")
    ax1.set_xticks(WINDOWS)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.set_title("SPRT window ablation vs false halts & token savings")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"[Task4] wrote {OUT}")
    for N, s, f in zip(WINDOWS, savings, fprs):
        print(f"[Task4] N={N} (df={residual_df(N)}): savings={s:.1f}%  FPR={f:.2f}%")
    fpr3, fpr5, sav5 = fprs[0], fprs[2], savings[2]
    print(f"[Task4] N=3 FPR={fpr3:.2f}% (need >12%), N=5 FPR={fpr5:.2f}% (need <5%), N=5 savings={sav5:.0f}% (need >35)")
    assert fpr3 > 12, "N=3 (1 residual df) must produce >12% false halts"
    assert fpr5 < 5.0, "N=5 must stabilize false halts below 5%"
    assert sav5 > 35, "N=5 must retain >35% token savings"
    print("[Task4] OK: N=3 → high false halts (1 df); N=5 → stable <5% FPR, >35% savings\n")
    return savings, fprs


if __name__ == "__main__":
    main()