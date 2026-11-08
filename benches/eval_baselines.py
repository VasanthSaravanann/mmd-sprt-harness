#!/usr/bin/env python3
"""WP2 — Baseline comparisons (fig8–fig9).

WP2.1 (fig8): Compares the full Gate 0 (3-gram RBF MMD²) against:
  - Linear MMD (3-gram, dot-product kernel)
  - Poly-2 MMD (3-gram, (1 + dot)² kernel)
  - KS statistic on the 3-gram frequency distribution
  - PSI (Population Stability Index) on the 3-gram distribution
  Reports FPR@95% TPR for each detector.

WP2.2 (fig9): OPE comparison — SNIPW, DR, DM estimators vs the IPW gate.
  Reports MAE and Spearman r_s against ground-truth viability rankings.
"""
from __future__ import annotations

import math
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from harness.evaluator import (
    N_EFF_MIN, R_MIN, P_MAX,
    effective_sample_size, winsorize,
    inverse_propensity_weight, assemble_combined,
    weighted_spearman_corr,
)
from harness.gate0 import (
    gate0, mmd2_3gram, normalized_ngram_vector,
    cosine_distance, mean_embedding, action_jaccard_distance,
    MMD2_FLOOR, D_ACT_THRESHOLD, SIGMA_BW,
)

OUT_ROC = os.path.join(os.path.dirname(__file__), "..", "paper", "figures", "fig8_baselines_roc.pdf")
OUT_BAR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures", "fig9_baselines_ope.pdf")
OUT_ROC = os.path.abspath(OUT_ROC)
OUT_BAR = os.path.abspath(OUT_BAR)

MOTIFS = [["a", "b", "b", "a"], ["a", "a", "b", "a"], ["b", "a", "a", "b"]]
SECRET = "S1"
N_TRAJECTORIES = 2000
N_STREAM = 90
TOKENS = 60


def motif_stream(length):
    m = random.choice(MOTIFS)
    return [m[i % len(m)] for i in range(length)]


def make_prod(seed):
    rng = random.Random(seed)
    return [motif_stream(TOKENS) for _ in range(N_STREAM)]


def healthy_cand(prod, seed):
    rng = random.Random(seed)
    return [list(rng.choice(["a", "b"]) if rng.random() < 0.002 else t for t in s) for s in prod]


def exfil_cand(prod, seed, rate=0.01):
    rng = random.Random(seed)
    out = []
    for s in prod:
        s = list(s)
        k = max(1, int(round(len(s) * rate)))
        for _ in range(k):
            s[rng.randrange(len(s))] = SECRET
        out.append(s)
    return out


# ---- Baseline detectors ----

def _spec_vecs(streams):
    """Return list of normalized 3-gram vectors as dicts."""
    return [normalized_ngram_vector(s) for s in streams]


def _vocab(streams_a, streams_b):
    va = set().union(*[set(s.keys()) for s in _spec_vecs(streams_a)])
    vb = set().union(*[set(s.keys()) for s in _spec_vecs(streams_b)])
    return sorted(va | vb)


def _mean_vec(streams, vocab):
    vs = _spec_vecs(streams)
    vec = np.zeros(len(vocab))
    for v in vs:
        for i, g in enumerate(vocab):
            vec[i] += v.get(g, 0.0)
    vec /= len(vs)
    return vec


def _linear_mmd(prod, cand):
    """3-gram linear (dot-product) MMD²: k(x,y) = <x,y>."""
    import math
    fx = _spec_vecs(cand)
    fy = _spec_vecs(prod)
    def lk(a, b):
        return sum(va * b.get(k, 0.0) for k, va in a.items())
    m, n = len(fx), len(fy)
    if m == 0 or n == 0:
        return float("inf")
    xx = sum(lk(fx[i], fx[j]) for i in range(m) for j in range(m)) / (m * m)
    yy = sum(lk(fy[i], fy[j]) for i in range(n) for j in range(n)) / (n * n)
    xy = sum(lk(fx[i], fy[j]) for i in range(m) for j in range(n)) / (m * n)
    return xx + yy - 2.0 * xy


def _poly2_mmd(prod, cand):
    """3-gram polynomial degree-2 MMD²: k(x,y) = (1 + <x,y>)²."""
    fx = _spec_vecs(cand)
    fy = _spec_vecs(prod)
    m, n = len(fx), len(fy)
    if m == 0 or n == 0:
        return float("inf")
    def pk(a, b):
        dot = sum(va * b.get(k, 0.0) for k, va in a.items())
        return (1.0 + dot) ** 2
    xx = sum(pk(fx[i], fx[j]) for i in range(m) for j in range(m)) / (m * m)
    yy = sum(pk(fy[i], fy[j]) for i in range(n) for j in range(n)) / (n * n)
    xy = sum(pk(fx[i], fy[j]) for i in range(m) for j in range(n)) / (m * n)
    return xx + yy - 2.0 * xy


def _ks_stat(prod, cand):
    """Two-sample KS statistic on the 3-gram frequency distribution.

    Builds the pooled 3-gram CDF from the mean embedding and computes the
    maximum absolute deviation between the production and candidate CDFs.
    This is a proper KS test (not just max|mean_diff|).
    """
    vocab = _vocab(prod, cand)
    mp = _mean_vec(prod, vocab)
    mc = _mean_vec(cand, vocab)
    # Sort vocab by gram for a valid CDF (use lexicographic order of grams)
    order = sorted(range(len(vocab)), key=lambda i: vocab[i])
    cdf_p = np.cumsum([mp[i] for i in order])
    cdf_c = np.cumsum([mc[i] for i in order])
    cdf_p /= cdf_p[-1] if cdf_p[-1] > 0 else 1.0
    cdf_c /= cdf_c[-1] if cdf_c[-1] > 0 else 1.0
    return float(np.max(np.abs(cdf_p - cdf_c)))


def _psi(prod, cand, bins=20):
    """Population Stability Index on the 3-gram mean embedding."""
    vocab = _vocab(prod, cand)
    mp = _mean_vec(prod, vocab) + 1e-12
    mc = _mean_vec(cand, vocab) + 1e-12
    return float(np.sum((mc - mp) * np.log(mc / mp)))


# ---- WP2.1 ----

def main():
    rng = random.Random(0)
    y_true = []
    scores = {name: [] for name in ["RBF MMD²", "linear MMD²", "poly-2 MMD²", "KS stat", "PSI", "cosine (mean emb)"]}

    for i in range(N_TRAJECTORIES):
        valid = rng.random() < 0.5
        prod = make_prod(i)
        rate = 0.0 if valid else 0.01 * (1 + rng.random())
        cand = healthy_cand(prod, i + 1000) if valid else exfil_cand(prod, i + 1000, rate=rate)

        y_true.append(1 if valid else 0)

        # RBF MMD² (current Gate 0)
        scores["RBF MMD²"].append(mmd2_3gram(cand, prod, SIGMA_BW))

        # Linear MMD (3-gram, dot product)
        scores["linear MMD²"].append(_linear_mmd(prod, cand))

        # Poly-2 MMD (3-gram)
        scores["poly-2 MMD²"].append(_poly2_mmd(prod, cand))

        # KS statistic (proper two-sample)
        scores["KS stat"].append(_ks_stat(prod, cand))

        # PSI
        scores["PSI"].append(_psi(prod, cand))

        # Cosine distance on mean embeddings (known to stay flat — fig1)
        scores["cosine (mean emb)"].append(
            cosine_distance(mean_embedding(cand), mean_embedding(prod)))

    y_true = np.array(y_true, int)

    # ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    results = []
    for name in ["RBF MMD²", "linear MMD²", "poly-2 MMD²", "KS stat", "PSI", "cosine (mean emb)"]:
        vals = np.array(scores[name])
        fpr, tpr, _ = roc_curve(1 - y_true, vals)
        a = auc(fpr, tpr)
        # FPR at 95% TPR
        idx_95 = np.searchsorted(tpr, 0.95)
        fpr_95 = fpr[min(idx_95, len(fpr) - 1)]
        results.append((name, a, fpr_95))
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={a:.3f}, FPR@95%={fpr_95:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("WP2.1 — Detector comparison (pos = exfiltrating)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_ROC), exist_ok=True)
    fig.savefig(OUT_ROC)
    plt.close(fig)
    print(f"[WP2.1] wrote {OUT_ROC}")
    for name, a, f95 in results:
        print(f"  {name:20s}  AUC={a:.3f}  FPR@95%TPR={f95:.3f}")

    # ---- WP2.2: OPE comparison ----
    print("\n[WP2.2] OPE comparison...")
    rng = np.random.default_rng(0)
    n = 2000
    # Features and ground-truth viability with heavy-tailed propensities.
    x = rng.uniform(0, 1, n)
    true_viability = 0.7 * x + 0.3 * rng.normal(0, 0.05, n)
    # Heavy-tailed propensities: most units have low selection probability,
    # a few have high probability (mimicking production IPW scenarios).
    propensities = np.clip(rng.pareto(2.0, n) * 0.1 + 0.05, 0.01, 0.95).tolist()
    outcomes = true_viability + rng.normal(0, 0.1, n)
    # Only 30% of units are "treated" (observed) — matches the 1% production subsample idea.
    treated = rng.binomial(1, np.array(propensities)) > 0
    y_obs = np.where(treated, outcomes, np.nan)
    obs_idx = np.where(treated)[0]
    n_obs = len(obs_idx)

    # IPW estimator on observed units
    weights = [inverse_propensity_weight(propensities[i]) for i in obs_idx]
    ww = winsorize(weights, 0.99)
    r_ipw, _ = weighted_spearman_corr(outcomes[obs_idx[:100]].tolist(), true_viability[obs_idx[:100]].tolist(), ww[:100])
    mae_ipw = float(np.mean(np.abs(
        np.array(ww[:100]) * np.array(outcomes[obs_idx[:100]]) / np.mean(ww[:100]) -
        true_viability[obs_idx[:100]]
    )))

    # SNIPW (Self-Normalized IPW) — correctly uses the normalized weights
    snipw_weights = np.array(ww) / np.mean(ww)
    r_snipw, _ = weighted_spearman_corr(outcomes[obs_idx[:100]].tolist(), true_viability[obs_idx[:100]].tolist(), snipw_weights[:100].tolist())
    mae_snipw = float(np.mean(np.abs(
        snipw_weights[:100] * np.array(outcomes[obs_idx[:100]]) - true_viability[obs_idx[:100]]
    )))

    # DM (Direct Method) — fit a linear model on the observed data
    from sklearn.linear_model import LinearRegression
    dm = LinearRegression()
    dm.fit(x[obs_idx[:100]].reshape(-1, 1), outcomes[obs_idx[:100]])
    y_dm = dm.predict(x[:100].reshape(-1, 1))
    r_dm, _ = weighted_spearman_corr(y_dm.tolist(), true_viability[:100].tolist(), [1.0] * 100)
    mae_dm = float(np.mean(np.abs(y_dm - true_viability[:100])))

    # DR (Doubly Robust) — IPW + DM correction
    y_cf = dm.predict(x[obs_idx[:100]].reshape(-1, 1)).flatten()
    dr_values = y_cf + np.array(ww[:100]) * (np.array(outcomes[obs_idx[:100]]) - y_cf) / np.mean(ww[:100])
    r_dr, _ = weighted_spearman_corr(dr_values.tolist(), true_viability[obs_idx[:100]].tolist(), ww[:100])
    mae_dr = float(np.mean(np.abs(dr_values - true_viability[obs_idx[:100]])))

    ope_results = [
        ("IPW", r_ipw, mae_ipw),
        ("SNIPW", r_snipw, mae_snipw),
        ("DM", r_dm, mae_dm),
        ("DR", r_dr, mae_dr),
    ]

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    names = [r[0] for r in ope_results]
    spearman_vals = [r[1] for r in ope_results]
    mae_vals = [r[2] for r in ope_results]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    ax1.bar(names, spearman_vals, color=colors)
    ax1.set_ylabel("Spearman r (higher is better)")
    ax1.set_title("OPE — rank correlation")
    ax1.axhline(0.85, color="gray", ls="--", lw=1)
    ax1.grid(alpha=0.3, axis="y")

    ax2.bar(names, mae_vals, color=colors)
    ax2.set_ylabel("MAE (lower is better)")
    ax2.set_title("OPE — mean absolute error")
    ax2.grid(alpha=0.3, axis="y")

    fig2.tight_layout()
    os.makedirs(os.path.dirname(OUT_BAR), exist_ok=True)
    fig2.savefig(OUT_BAR)
    plt.close(fig2)
    print(f"[WP2.2] wrote {OUT_BAR}")
    for name, r, mae in ope_results:
        print(f"  {name:10s}  Spearman r={r:.3f}  MAE={mae:.4f}")


if __name__ == "__main__":
    main()