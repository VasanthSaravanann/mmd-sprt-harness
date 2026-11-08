#!/usr/bin/env python3
"""Task 3 — IPW variance stabilization & weight winsorization sweep.

Samples inverse-propensity weights from heavy-tailed Pareto distributions across
a shape-parameter gradient alpha in [1.1, 3.0], then reports the effective
sample size n_eff = (sum w)^2 / sum w^2 for RAW vs P99-winsorized weights
(w_i^cap = min(w_i, P99(w))).  Also computes the weighted Spearman IPW rank
correlation against a known ground-truth viability ranking.

Renders fig3_ipw_winsorization.pdf (alpha vs n_eff, highlighting n_eff >= 30).

Assertion: unwinsorized weights collapse n_eff < 30, while P99 winsorization
preserves statistical power (n_eff >= 30) up to alpha = 1.2.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from harness.evaluator import effective_sample_size, winsorize  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "figures", "fig3_ipw_winsorization.pdf")
OUT = os.path.abspath(OUT)

N_SAMPLES = 420
TRIALS = 25
ALPHAS = np.round(np.arange(1.1, 3.01, 0.1), 2)


def weighted_spearman_r(y, weights):
    """Weighted Spearman rank correlation (ranks + weighted Pearson)."""
    rx = rankdata(np.arange(len(y)))
    ry = rankdata(y)
    x = rx.astype(float)
    yy = ry.astype(float)
    w = np.asarray(weights, float)
    wsum = w.sum()
    mx = (x * w).sum() / wsum
    my = (yy * w).sum() / wsum
    cov = (w * (x - mx) * (yy - my)).sum() / wsum
    vx = (w * (x - mx) ** 2).sum() / wsum
    vy = (w * (yy - my) ** 2).sum() / wsum
    if vx == 0 or vy == 0:
        return 0.0
    return cov / np.sqrt(vx * vy)


def main():
    raw_means, win_means = [], []
    for a in ALPHAS:
        raw_acc, win_acc = [], []
        for t in range(TRIALS):
            w = np.random.pareto(a, N_SAMPLES).tolist()
            raw_acc.append(effective_sample_size(w))
            win_acc.append(effective_sample_size(winsorize(w, 0.99)))
        raw_means.append(np.mean(raw_acc))
        win_means.append(np.mean(win_acc))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ALPHAS, raw_means, marker="o", ms=4, color="tab:red", label="raw IPW weights")
    ax.plot(ALPHAS, win_means, marker="s", ms=4, color="tab:green", label="P99-winsorized weights")
    ax.axhline(30, color="gray", ls="--", lw=1.3, label="n_eff >= 30 gate")
    ax.axvspan(1.1, 1.2, color="gold", alpha=0.15, label="heavy-tail regime α ≤ 1.2")
    ax.set_xlabel("Pareto tail-skewness α")
    ax.set_ylabel("effective sample size n_eff")
    ax.set_title("IPW P99 winsorization stabilizes n_eff under heavy tails")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"[Task3] wrote {OUT}")

    # weighted Spearman against a known viability ranking (auxiliary gate check)
    w = np.random.pareto(1.2, N_SAMPLES).tolist()
    wc = winsorize(w, 0.99)
    true_rank = np.arange(N_SAMPLES)
    aligned = true_rank + np.random.normal(0, 4.0, N_SAMPLES)
    shuffled = np.random.permutation(true_rank)
    r_aligned = weighted_spearman_r(aligned, wc)
    r_shuffled = weighted_spearman_r(shuffled, wc)

    heavy = ALPHAS <= 1.2
    raw_min = min(raw_means[i] for i in range(len(ALPHAS)) if heavy[i])
    win_min = min(win_means[i] for i in range(len(ALPHAS)) if heavy[i])
    print(f"[Task3] raw n_eff min (α≤1.2)={raw_min:.1f} (must be <30)")
    print(f"[Task3] winsorized n_eff min (α≤1.2)={win_min:.1f} (must be >=30)")
    print(f"[Task3] weighted-Spearman r aligned={r_aligned:.3f} (must be >0.85), shuffled={r_shuffled:.3f}")
    assert raw_min < 30, f"raw weights must collapse below 30, got {raw_min:.1f}"
    assert win_min >= 30, f"winsorized must preserve power >=30, got {win_min:.1f}"
    assert r_aligned > 0.85 and r_shuffled < 0.85
    print("[Task3] OK: raw collapses; P99 winsorization preserves n_eff >= 30 and IPW correlation is strong\n")
    return raw_means, win_means


if __name__ == "__main__":
    main()