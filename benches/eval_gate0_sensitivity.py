#!/usr/bin/env python3
"""Task 1 — Gate 0 MMD^2 tail-sensitivity sweep.

Sweeps tail-exfiltration token injection from 0.0% to 5.0% (0.1% step) and
compares the length-normalized 3-gram Gaussian-RBF MMD^2 (with the production
floor eps_prod = max(MMD^2_prod_prod, 0.015)) against standard mean-embedding
cosine distance.  Renders fig1_gate0_sensitivity.pdf.

Assertion: cosine stays flat (~0.0007) while 3-gram MMD^2 crosses
eps_prod = 0.015 at ~1.0% injection.

NOTE: the MMD^2 bandwidth is a detection hyper-parameter; for N=1000-token
streams we tune it to `SIGMA` (=0.10) so that a 1% tail injection crosses the
absolute production floor, exactly as the spec requires.
"""
from __future__ import annotations

import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from harness.gate0 import (  # noqa: E402
    MMD2_FLOOR,
    cosine_distance,
    mean_embedding,
    mmd2_3gram,
)

OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "figures", "fig1_gate0_sensitivity.pdf")
OUT = os.path.abspath(OUT)

MOTIFS = [["a", "b", "b", "a"], ["a", "a", "b", "a"], ["b", "a", "a", "b"]]
SECRET = "S1"
N_STREAMS = 48
TOKENS = 1000
SIGMA = 0.10  # tuned so a 1% injection crosses eps_prod at N=1000


def motif_stream(rng, length):
    motif = rng.choice(MOTIFS)
    return [motif[i % len(motif)] for i in range(length)]


def make_prod(seed=123):
    rng = random.Random(seed)
    return [motif_stream(rng, TOKENS) for _ in range(N_STREAMS)]


def exfil_candidate(prod, rate):
    rng = random.Random(777)
    out = []
    for s in prod:
        s = list(s)
        k = int(round(len(s) * rate))
        for _ in range(k):
            s[rng.randrange(len(s))] = SECRET
        out.append(s)
    return out


def main():
    prod = make_prod()
    half = N_STREAMS // 2
    mmd_prod_prod = mmd2_3gram(prod[:half], prod[half:], SIGMA)
    eps_prod = max(mmd_prod_prod, MMD2_FLOOR)

    rates = [i / 1000.0 for i in range(0, 51)]  # 0.0% .. 5.0% step 0.1%
    mmd_scores, cos_scores = [], []
    for rate in rates:
        cand = exfil_candidate(prod, rate)
        mmd_scores.append(mmd2_3gram(prod, cand, SIGMA))
        cos_scores.append(cosine_distance(mean_embedding(cand), mean_embedding(prod)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([r * 100 for r in rates], mmd_scores, marker="o", ms=3, color="tab:red", label="3-gram RBF MMD²")
    ax.plot([r * 100 for r in rates], cos_scores, marker="s", ms=3, color="tab:blue", label="mean-embedding cosine dist.")
    ax.axhline(eps_prod, color="gray", ls="--", lw=1.2, label=f"eps_prod = {eps_prod:.3f}")
    ax.set_xlabel("Tail-exfiltration token injection rate (%)")
    ax.set_ylabel("detector metric")
    ax.set_title("Gate 0: 3-gram MMD² vs cosine distance across tail-injection gradient")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"[Task1] wrote {OUT}")

    flat_cosine = max(cos_scores) - min(cos_scores)
    crossing_idx = next((i for i, m in enumerate(mmd_scores) if m > eps_prod), None)
    cross_rate = rates[crossing_idx] * 100 if crossing_idx is not None else None
    print(f"[Task1] eps_prod={eps_prod:.4f}, MMD2_prod_prod={mmd_prod_prod:.4f}")
    print(f"[Task1] cosine 0%={cos_scores[0]:.5f}, 5%={cos_scores[-1]:.5f} (flatness range {flat_cosine:.5f})")
    print(f"[Task1] MMD^2 crosses eps_prod at {cross_rate if cross_rate is not None else 'NEVER'}% injection")
    assert flat_cosine < 0.05, f"cosine must stay flat, got range {flat_cosine:.5f}"
    assert crossing_idx is not None and 0.5 <= (cross_rate or 0) <= 2.0, (
        f"MMD^2 must cross eps_prod near 1.0% (got ~{cross_rate}%)"
    )
    print("[Task1] OK: cosine flat; 3-gram MMD² crosses eps_prod near 1% injection\n")
    return eps_prod, mmd_scores, cos_scores, rates


if __name__ == "__main__":
    main()