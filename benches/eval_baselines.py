#!/usr/bin/env python3
"""WP2 — Baseline comparisons (fig8–fig9).

Compares the full Gate 0 (3-gram RBF MMD² + Action Jaccard) against simpler
detectors: cosine-only, frequency-vector-only, and naive-Bayes-on-3-grams.
Renders two figures:
  fig8_baselines_roc.pdf — ROC curves for each detector
  fig9_baselines_auroc.pdf — bar chart of AUROC for each detector
"""
from __future__ import annotations

import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from harness.gate0 import gate0, mmd2_3gram, cosine_distance, mean_embedding, action_jaccard_distance, normalized_ngram_vector

OUT_ROC = os.path.join(os.path.dirname(__file__), "..", "paper", "figures", "fig8_baselines_roc.pdf")
OUT_BAR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures", "fig9_baselines_auroc.pdf")
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


def main():
    rng = random.Random(0)
    y_true = []
    mmd_scores = []
    cos_scores = []
    dact_scores = []
    # combined Gate 0 soft score: 1 - max(normalized mmd, normalized d_act)
    gate0_scores = []

    for i in range(N_TRAJECTORIES):
        valid = rng.random() < 0.5
        prod = make_prod(i)
        cand = healthy_cand(prod, i + 1000) if valid else exfil_cand(prod, i + 1000, rate=0.01 * (1 + rng.random()))

        g = gate0(prod, cand)
        y_true.append(1 if valid else 0)
        mmd_scores.append(g.mmd2)
        cos_scores.append(g.cosine_dist)
        dact_scores.append(g.d_act)

    y_true = np.array(y_true, int)

    detectors = [
        ("MMD² (3-gram RBF)", mmd_scores),
        ("cosine distance", cos_scores),
        ("Action Jaccard D_act", dact_scores),
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = []
    aurocs = []
    for name, scores in detectors:
        fpr, tpr, _ = roc_curve(1 - y_true, scores)  # positive = exfil (1)
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={a:.3f})")
        labels.append(name)
        aurocs.append(a)

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Detector ROC curves (pos = exfiltrating)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_ROC), exist_ok=True)
    fig.savefig(OUT_ROC)
    plt.close(fig)
    print(f"[WP2] wrote {OUT_ROC}")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    ax2.bar(labels, aurocs, color=colors)
    ax2.set_ylabel("AUROC")
    ax2.set_title("Baseline detector comparison")
    ax2.set_ylim(0, 1.05)
    for i, v in enumerate(aurocs):
        ax2.text(i, v + 0.02, f"{v:.3f}", ha="center")
    ax2.grid(alpha=0.3, axis="y")
    fig2.tight_layout()
    fig2.savefig(OUT_BAR)
    plt.close(fig2)
    print(f"[WP2] wrote {OUT_BAR}")


if __name__ == "__main__":
    main()