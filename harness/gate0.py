"""Gate 0: distributional attestation engine.

The gate compares the *candidate* stream distribution against the locked
*production* distribution using a length-normalized 3-gram MMD-squared
statistic with a production-ground floor

    eps_prod = max(MMD2_prod_prod, 0.015)

and an Action Jaccard distance test :math:`D_{act} < 0.05`.

A crucial property exercised by the tests: a 1% *tail-exfiltration* injection
(rare tokens withheld from training then surfaced in the candidate) can keep
the *mean-embedding cosine distance* small while driving the 3-gram MMD-squared
above ``eps_prod``.  Gate 0 therefore fails on exfiltration even when a
cosine-only detector would pass -- this is the raison d'etre of the MMD floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

MMD2_FLOOR = 0.015
D_ACT_THRESHOLD = 0.05
NGRAM_N = 3
# Gaussian RBF bandwidth for the n-gram MMD kernel.  Production telemetry in
# a mature system is highly structured/low-entropy, so a tight bandwidth keeps
# the kernel sensitive to small 3-gram tail perturbations (1% exfiltration)
# while intra-production dispersion stays well below the MMD floor.
SIGMA_BW = 0.15

# Sentinel used to zero-pad streams shorter than `n` tokens (mirrors the Rust
# `[u8; NGRAM_N]` zero-padding so both implementations agree exactly).
PAD_TOKEN = "\x00"


# --------------------------------------------------------------------------
# Tokenization / feature helpers
# --------------------------------------------------------------------------
def ngrams(tokens: Sequence[str], n: int = NGRAM_N) -> List[tuple]:
    """Extract overlapping character-level `n`-grams over token labels.

    Streams shorter than ``n`` are zero-padded with :data:`PAD_TOKEN` and emit a
    single gram, matching the Rust `spectrum` implementation so cross-language
    results agree exactly.  (Previous behaviour emitted a mix of bare tokens and
    tuples for short streams, which made the action set inconsistent.)
    """
    if len(tokens) < n:
        padded = tuple(list(tokens) + [PAD_TOKEN] * (n - len(tokens)))
        return [padded]
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def normalized_ngram_vector(tokens: Sequence[str], n: int = NGRAM_N) -> Dict[tuple, float]:
    """Length-normalized n-gram count vector (each stream == a distribution)."""
    grams = ngrams(tokens, n)
    total = len(grams) or 1
    counts: Dict[tuple, float] = {}
    for g in grams:
        counts[g] = counts.get(g, 0.0) + 1.0
    return {g: c / total for g, c in counts.items()}


def _rbf_kernel(a: Dict[tuple, float], b: Dict[tuple, float],
                sigma: float = SIGMA_BW) -> float:
    """Gaussian RBF kernel over length-normalized n-gram features."""
    keys = set(a) | set(b)
    dist2 = sum((a.get(k, 0.0) - b.get(k, 0.0)) ** 2 for k in keys)
    return math.exp(-dist2 / (2.0 * sigma * sigma))


def mmd2_3gram(x_streams: Sequence[Sequence[str]],
               y_streams: Sequence[Sequence[str]],
               sigma: float = SIGMA_BW) -> float:
    """Biased/quadratic MMD-squared estimate over 3-gram RBF features.

    ``MMD^2 = E[k(x,x')] + E[k(y,y')] - 2 E[k(x,y)]`` with :math:`k` the Gaussian
    RBF kernel on length-normalized n-gram vectors.
    """
    fx = [normalized_ngram_vector(s) for s in x_streams]
    fy = [normalized_ngram_vector(s) for s in y_streams]
    m, n = len(fx), len(fy)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return math.inf

    term_xx = sum(_rbf_kernel(fx[i], fx[j], sigma) for i in range(m) for j in range(m)) / (m * m)
    term_yy = sum(_rbf_kernel(fy[i], fy[j], sigma) for i in range(n) for j in range(n)) / (n * n)
    term_xy = sum(_rbf_kernel(fx[i], fy[j], sigma) for i in range(m) for j in range(n)) / (m * n)
    return term_xx + term_yy - 2.0 * term_xy


def mean_embedding(streams: Sequence[Sequence[str]], n: int = NGRAM_N) -> Dict[tuple, float]:
    """Mean over per-stream normalized n-gram vectors (used for cosine checks)."""
    vecs = [normalized_ngram_vector(s, n) for s in streams]
    if not vecs:
        return {}
    keys = set().union(*vecs) if vecs else set()
    out: Dict[tuple, float] = {}
    for k in keys:
        out[k] = sum(v.get(k, 0.0) for v in vecs) / len(vecs)
    return out


def _dict_norm(v: Dict[tuple, float]) -> float:
    return math.sqrt(sum(w * w for w in v.values()))


def cosine_distance(a: Dict[tuple, float], b: Dict[tuple, float]) -> float:
    dot = sum(w * b.get(k, 0.0) for k, w in a.items())
    na, nb = _dict_norm(a), _dict_norm(b)
    if na == 0.0 or nb == 0.0:
        return 1.0
    return 1.0 - (dot / (na * nb))


# --------------------------------------------------------------------------
# Action Jaccard distance
# --------------------------------------------------------------------------
def _action_set(streams: Sequence[Sequence[str]], n: int = NGRAM_N) -> set:
    out: set = set()
    for s in streams:
        out.update(ngrams(s, n))
    return out


def action_jaccard_distance(x_streams: Sequence[Sequence[str]],
                            y_streams: Sequence[Sequence[str]]) -> float:
    a, b = _action_set(x_streams), _action_set(y_streams)
    union = a | b
    if not union:
        return 0.0
    inter = a & b
    return 1.0 - (len(inter) / len(union))


# --------------------------------------------------------------------------
# Gate decision
# --------------------------------------------------------------------------
@dataclass
class Gate0Result:
    mmd2: float
    eps_prod: float
    d_act: float
    cosine_dist: float
    status: str
    passed: bool
    reason: str


def gate0(prod_streams: Sequence[Sequence[str]],
          cand_streams: Sequence[Sequence[str]]) -> Gate0Result:
    """Run the Gate 0 attestation on candidate vs production streams.

    Degenerate production sets are guarded: with 0 or 1 production streams the
    within-production dispersion cannot be estimated (the second split half is
    empty and MMD^2 is +inf), which previously set ``eps_prod = inf`` and let
    ANY candidate pass the distributional check.  In that case we use the
    absolute floor ``MMD2_FLOOR`` as ``eps_prod``.
    """
    if len(prod_streams) < 2:
        mmd_prod_prod = 0.0
        eps_prod = MMD2_FLOOR
    else:
        # Production dispersion: split into two halves and measure within-prod MMD.
        half = max(1, len(prod_streams) // 2)
        prod_a = prod_streams[:half]
        prod_b = prod_streams[half:]
        mmd_prod_prod = mmd2_3gram(prod_a, prod_b)
        eps_prod = max(mmd_prod_prod, MMD2_FLOOR)

    mmd2 = mmd2_3gram(cand_streams, prod_streams)
    d_act = action_jaccard_distance(cand_streams, prod_streams)

    prod_emb = mean_embedding(prod_streams)
    cand_emb = mean_embedding(cand_streams)
    cosine = cosine_distance(cand_emb, prod_emb)

    dist_ok = mmd2 <= eps_prod
    act_ok = d_act < D_ACT_THRESHOLD
    passed = bool(dist_ok and act_ok)

    if not dist_ok:
        status = "GATE0_FAIL"
        reason = (
            f"GATE0_FAIL: MMD^2={mmd2:.4f} > eps_prod={eps_prod:.4f} "
            "(distribution shift / tail exfiltration)"
        )
    elif not act_ok:
        status = "GATE0_FAIL"
        reason = f"GATE0_FAIL: Action Jaccard D_act={d_act:.4f} >= 0.05"
    else:
        status = "GATE0_PASS"
        reason = "GATE0_PASS: distributionally aligned with production"
    return Gate0Result(mmd2, eps_prod, d_act, cosine, status, passed, reason)