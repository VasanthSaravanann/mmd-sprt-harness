#!/usr/bin/env python3
"""Measure a NumPy baseline for the RBF-MMD^2 kernel (Task 7 comparison).

Builds dense length-normalized 3-gram spectra as feature matrices and computes
the biased Gaussian-RBF MMD^2 in NumPy, timing the per-call cost.  Prints the
NumPy calls/sec so the Rust adjacency (>=15x) can be evaluated.
"""
import random
import time
from collections import Counter

import numpy as np

MOTIFS = [["a", "b", "b", "a"], ["a", "a", "b", "a"], ["b", "a", "a", "b"]]
N_STREAM = 48
LEN = 64


def spectrum(s):
    c = Counter(tuple(s[i:i + 3]) for i in range(max(1, len(s) - 2)))
    total = sum(c.values()) or 1
    return {k: v / total for k, v in c.items()}


def streams(n, ln):
    return [[random.choice(MOTIFS)[i % 4] for i in range(ln)] for _ in range(n)]


def main():
    random.seed(0)
    A = streams(N_STREAM, LEN)
    B = streams(N_STREAM, LEN)
    vocab = sorted(set().union(*[set(spectrum(s)) for s in A + B]))
    idx = {k: i for i, k in enumerate(vocab)}

    def to_mat(S):
        M = np.zeros((len(S), len(vocab)))
        for i, s in enumerate(S):
            for k, v in spectrum(s).items():
                M[i, idx[k]] = v
        return M

    MA, MB = to_mat(A), to_mat(B)

    def mmd_np(X, Y, sigma=0.10):
        Gxx = np.inner(X, X)
        dx = np.einsum("ii->i", Gxx)[:, None]
        Kxx = np.exp(-(dx + dx.T - 2 * Gxx) / (2 * sigma * sigma))
        Gyy = np.inner(Y, Y)
        dy = np.einsum("ii->i", Gyy)[None, :]
        Kyy = np.exp(-(dy + dy.T - 2 * Gyy) / (2 * sigma * sigma))
        Gxy = np.inner(X, Y)
        d2 = np.einsum("ii->i", Gxx)[:, None] + np.einsum("ii->i", Gyy)[None, :] - 2 * Gxy
        Kxy = np.exp(-d2 / (2 * sigma * sigma))
        return float(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean())

    for _ in range(5):
        mmd_np(MA, MB)
    t0 = time.perf_counter()
    for _ in range(40):
        mmd_np(MA, MB)
    dt = (time.perf_counter() - t0) / 40
    calls = 1.0 / dt
    us = dt * 1e6
    print(f"[numpy baseline] mmd2 per call = {us:.0f} us  ({calls:.1f} calls/sec)")
    import json
    json.dump({"numpy_mmd2_usec_per_call": us, "numpy_mmd2_calls_per_sec": calls},
              open("benches/numpy_baseline.json", "w"), indent=2)
    print("[numpy baseline] wrote benches/numpy_baseline.json")


if __name__ == "__main__":
    main()