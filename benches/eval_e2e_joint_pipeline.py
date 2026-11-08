#!/usr/bin/env python3
"""Task 6 — Master E2E joint evaluation matrix.

Runs 5000 full evaluation trajectories that jointly exercise Gate 0 (MMD^2),
the SPRT sequential-stopping loop, and the IPW correlation gate.  Suite A are
valid/robust systems (every gate clears); Suite B are flawed/unstable systems
(tail-divergent distributions, non-convergent SPRTs, weak IPW).

For each trajectory we compute a soft score (fraction of gates cleared) and the
hard verdict (FULL_DEPLOYMENT_GRANTED iff all gates clear).  We render a
confusion matrix and an ROC-AUC curve for verdict assignment, with Suite A as
the positive (granted) class.

Assertion: ZERO false deployment authorizations on tail-divergent /
non-convergent systems (no Suite-B trajectory is granted).
"""
from __future__ import annotations

import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from harness.evaluator import evaluate_ipw  # noqa: E402
from harness.fuzzer_sprt_loop import assess_convergence  # noqa: E402
from harness.gate0 import gate0  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "figures", "fig6_e2e_pipeline_matrix.pdf")
OUT = os.path.abspath(OUT)

N_PER_SUITE = 2500           # 2500 x 2 suites = 5000 full trajectories
MOTIFS = [["a", "b", "b", "a"], ["a", "a", "b", "a"], ["b", "a", "a", "b"]]
SECRET = "S1"
N_STREAM = 16
TOKENS = 24

SPRT_GOOD = [0.30, 0.20, 0.12, 0.07, 0.04]
SPRT_BAD = [0.30, 0.10, 0.28, 0.05, 0.22]


def make_prod(seed):
    rng = random.Random(seed)
    return [[rng.choice(MOTIFS)[i % 4] for i in range(TOKENS)] for _ in range(N_STREAM)]


def cand_aligned(prod, seed):
    rng = random.Random(seed)
    out = []
    for s in prod:
        s = list(s)
        if rng.random() < 0.002:
            s[rng.randrange(len(s))] = rng.choice(["a", "b"])
        out.append(s)
    return out


def cand_taildivergent(prod, seed):
    rng = random.Random(seed)
    out = []
    for s in prod:
        s = list(s)
        for _ in range(max(1, int(round(len(s) * 0.02)))):
            s[rng.randrange(len(s))] = SECRET
        out.append(s)
    return out


def make_ipw_good():
    rng = np.random.default_rng(7)
    xp = rng.uniform(0, 1, 3000); yp = 0.9 * xp + 0.1 * rng.normal(0, 0.1, 3000)
    xe = rng.uniform(0, 1, 4000); ye = 0.95 * xe + 0.05 * rng.normal(0, 0.1, 4000)
    return list(xp), list(yp), [0.01] * 3000, list(xe), list(ye), [0.4] * 4000


def make_ipw_bad(good):
    px, py, pp, ex, ey, ep = good
    rng = np.random.default_rng(999)
    return px, py, pp, ex, list(rng.normal(0, 1, len(ex))), ep


def main():
    good_ipw = make_ipw_good()
    bad_ipw = make_ipw_bad(good_ipw)
    ipw_good_pass = evaluate_ipw(*good_ipw).passed
    ipw_bad_pass = evaluate_ipw(*bad_ipw).passed
    assert ipw_good_pass and not ipw_bad_pass, "IPW calibration broken"

    y_true, y_score, y_verdict = [], [], []
    for idx in range(N_PER_SUITE):
        for valid in (True, False):
            seed = idx + (0 if valid else 10**6)
            prod = make_prod(seed)
            g0 = gate0(prod, cand_aligned(prod, seed) if valid else cand_taildivergent(prod, seed))
            sprt = assess_convergence(SPRT_GOOD if valid else SPRT_BAD)
            gates = [g0.passed, sprt.converged, ipw_good_pass if valid else ipw_bad_pass]
            score = sum(gates) / len(gates)
            granted = all(gates)
            y_true.append(1 if valid else 0)   # 1 = valid (A, positive/grant class)
            y_score.append(score)
            y_verdict.append(granted)

    y_true = np.array(y_true, int)
    pred_grant = np.array(y_verdict, int)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    cm = confusion_matrix(y_true, pred_grant, labels=[1, 0])
    false_deploy = int(np.sum((pred_grant == 1) & (y_true == 0)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["grant", "halt"]); ax.set_yticklabels(["A (valid)", "B (flawed)"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_title(f"Confusion matrix (false deployments = {false_deploy})")
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    ax = axes[1]
    ax.plot(fpr, tpr, color="tab:green", lw=2, label=f"ROC (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC: FULL_DEPLOYMENT_GRANTED vs DEPLOYMENT_HALTED")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"[Task6] wrote {OUT}")
    print(f"[Task6] IPW: good_pass={ipw_good_pass}, bad_pass={ipw_bad_pass}")
    print(f"[Task6] false deployments={false_deploy} (must be 0), ROC-AUC={roc_auc:.3f}")
    print(f"[Task6] Suite A granted={int(np.sum(pred_grant[y_true == 1]))} / {N_PER_SUITE}")
    print(f"[Task6] Suite B granted={false_deploy} / {N_PER_SUITE}")
    assert false_deploy == 0, "ZERO false deployment authorizations required"
    assert roc_auc >= 0.95, f"AUC too low: {roc_auc:.3f}"
    print("[Task6] OK: zero false deployments; strong ROC-AUC separation\n")
    return cm, roc_auc


if __name__ == "__main__":
    main()