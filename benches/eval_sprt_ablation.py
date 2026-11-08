#!/usr/bin/env python3
"""Task 4 — SPRT rolling-fit-window (N) ablation study.

Executes `N_TRIALS` stochastic healthy trajectories across rolling-window sizes
N in {3, 4, 5, 6, 7}.  For each N it performs a REAL exponential-decay fit
`d^(i) = beta0 e^(-lambda i)` via `harness.fuzzer_sprt_loop.fit_exponential_decay`
and records:

  * false-decision rate — the fraction of healthy trajectories whose window-N
    verdict disagrees with the full-window (N=7) ground-truth verdict.  Short
    windows overfit a handful of noisy points and mis-decide; this is the
    "false halt / false pass" rate introduced by an under-powered fit.
  * token savings — `(K_MAX - N) / K_MAX` for early stopping at the first window.

All numbers are MEASURED from real fits (previously this benchmark sampled a
hand-tuned Gaussian calibrated to produce the wanted answer).

Renders fig4_sprt_window_ablation.pdf.

Assertion: N=3 false-decision rate > 12% (1 residual df), while N=5 drops below
5% and retains > 35% token savings.
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

from harness.fuzzer_sprt_loop import fit_exponential_decay  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "figures", "fig4_sprt_window_ablation.pdf")
OUT = os.path.abspath(OUT)

N_TRIALS = 2000
K_MAX = 8
WINDOWS = [3, 4, 5, 6, 7]
GROUND_TRUTH_WINDOW = 7
EARLY_NOISE = 0.30  # relative noise, large early (few samples) and decaying


def healthy_trajectory(rng):
    """A healthy, exponentially-converging trajectory with early-heavy noise."""
    d0 = rng.uniform(0.3, 0.5)
    lam = rng.uniform(0.5, 0.8)
    out = []
    for i in range(K_MAX):
        clean = d0 * math.exp(-lam * i)
        sigma = EARLY_NOISE / (1.0 + i)  # noise shrinks as samples accumulate
        out.append(max(clean * (1.0 + rng.gauss(0.0, sigma)), 1e-4))
    return out


def window_verdict(dist, n):
    """Real exponential-decay fit over the first `n` rounds -> converged?"""
    return fit_exponential_decay(dist[:n]).converged


def main():
    rng = random.Random(0)
    trajectories = [healthy_trajectory(rng) for _ in range(N_TRIALS)]
    # full-window ground truth per trajectory
    truth = [window_verdict(d, GROUND_TRUTH_WINDOW) for d in trajectories]

    fprs, savings = [], []
    for n in WINDOWS:
        disagree = 0
        for d, t in zip(trajectories, truth):
            if window_verdict(d, n) != t:
                disagree += 1
        fprs.append(disagree / N_TRIALS * 100)
        savings.append((1.0 - n / K_MAX) * 100)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(WINDOWS, savings, marker="o", ms=6, color="tab:blue", label="token savings (%)")
    ax2.plot(WINDOWS, fprs, marker="s", ms=6, color="tab:red", label="false-decision rate (%)")
    ax1.axhline(35, color="tab:blue", ls=":", lw=1.1)
    ax2.axhline(5.0, color="tab:red", ls=":", lw=1.1)
    ax1.set_xlabel("rolling fit window N  (residual df = N − 2)")
    ax1.set_ylabel("token savings (%)", color="tab:blue")
    ax2.set_ylabel("false-decision rate (%)", color="tab:red")
    ax1.set_xticks(WINDOWS)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.set_title("SPRT window ablation (real exponential fits)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)

    print(f"[Task4] wrote {OUT}")
    for n, s, f in zip(WINDOWS, savings, fprs):
        print(f"[Task4] N={n} (df={max(n - 2, 1)}): savings={s:.1f}%  false-decision={f:.2f}%")
    f3, f5, s5 = fprs[0], fprs[2], savings[2]
    print(f"[Task4] N=3 FDR={f3:.2f}% (need >12), N=5 FDR={f5:.2f}% (need <5), N=5 savings={s5:.0f}% (need >35)")
    assert f3 > 12, f"N=3 must mis-decide >12% of the time (1 residual df), got {f3:.1f}%"
    assert f5 < 5.0, f"N=5 must stabilize the decision below 5%, got {f5:.1f}%"
    assert s5 > 35, f"N=5 must retain >35% token savings, got {s5:.0f}%"
    print("[Task4] OK: N=3 → high false-decision rate; N=5 → <5% with >35% savings\n")
    return savings, fprs


if __name__ == "__main__":
    main()