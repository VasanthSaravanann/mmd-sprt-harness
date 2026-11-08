"""Task 2: Gate 0 distributional attestation (MMD^2 floor + Action Jaccard)."""

import random

import pytest

from harness.gate0 import (
    D_ACT_THRESHOLD,
    MMD2_FLOOR,
    SIGMA_BW,
    action_jaccard_distance,
    cosine_distance,
    gate0,
    mean_embedding,
    mmd2_3gram,
    normalized_ngram_vector,
)

MOTIFS = [["a", "b", "b", "a"], ["a", "a", "b", "a"], ["b", "a", "a", "b"]]
SECRET = "S1"  # exfiltration / withheld tail token


def _motif_stream(motif, length=60):
    return [motif[i % len(motif)] for i in range(length)]


def _make_production(seed=123, n=90):
    rng = random.Random(seed)
    return [_motif_stream(rng.choice(MOTIFS)) for _ in range(n)]


def _healthy_candidate(prod, seed=124):
    """Robust-system output: reproduces production motifs; only ~0.2% jitter."""
    rng = random.Random(seed)
    out = []
    for s in prod:
        s = list(s)
        k = int(round(len(s) * 0.002))
        for _ in range(k):
            s[rng.randrange(len(s))] = rng.choice(["a", "b"])
        out.append(s)
    return out


def _exfil_candidate(prod, seed=125, fraction=0.01):
    """Candidate = production with ~1% tail-exfiltration token injection."""
    rng = random.Random(seed)
    out = []
    for s in prod:
        s = list(s)
        k = max(1, int(round(len(s) * fraction)))
        for _ in range(k):
            s[rng.randrange(len(s))] = SECRET
        out.append(s)
    return out


def test_mmd2_floor_constant():
    assert MMD2_FLOOR == 0.015


def test_action_threshold():
    assert D_ACT_THRESHOLD == 0.05


def test_length_normalization_produces_distributions():
    short = normalized_ngram_vector(["a", "b", "c"])
    long = normalized_ngram_vector(["a", "b", "c"] * 9)
    # Both are proper probability distributions over their 3-gram repertoire.
    assert sum(short.values()) == pytest.approx(1.0)
    assert sum(long.values()) == pytest.approx(1.0)
    # Longer streams use the SAME recurring n-gram set, just proportionally more
    # of it -- so the normalized weight of a fully-recurring gram stays ~1 for
    # short streams (1 gram) and is still normalized for long ones.
    assert ("a", "b", "c") in short


def test_mmd2_zero_for_identical_streams():
    s = ["a", "b", "c", "a", "b", "c"]
    assert mmd2_3gram([s, s], [s, s]) == pytest.approx(0.0, abs=1e-9)


def test_gate0_passes_on_healthy_candidate():
    prod = _make_production()
    res = gate0(prod, _healthy_candidate(prod))
    assert res.passed is True, res.reason
    assert res.mmd2 <= res.eps_prod
    assert res.d_act < D_ACT_THRESHOLD


def test_gate0_fails_on_1_percent_tail_exfiltration():
    prod = _make_production()
    res = gate0(prod, _exfil_candidate(prod))
    assert res.passed is False, "MMD^2 floor must catch 1% tail exfiltration"
    assert res.mmd2 > res.eps_prod, (
        f"MMD^2={res.mmd2:.4f} must exceed eps_prod={res.eps_prod:.4f}"
    )


def test_exfiltration_evades_cosine_but_not_mmd():
    prod = _make_production()
    cand = _exfil_candidate(prod)

    cosine = cosine_distance(mean_embedding(cand), mean_embedding(prod))
    res1 = gate0(prod, _healthy_candidate(prod))
    res2 = gate0(prod, cand)

    # A cosine-only / mean-embedding detector "passes" the exfil candidate...
    assert abs(cosine) < 0.05
    assert res1.passed is True
    # ...but the 3-gram MMD^2 anchored at eps_prod flips the gate to FAIL.
    assert res2.passed is False
    assert res2.mmd2 > res2.eps_prod


def test_normal_candidate_uses_shared_actions():
    # Healthy content stays within the production action vocabulary.
    prod = _make_production()
    cand = _healthy_candidate(prod)
    assert action_jaccard_distance(cand, prod) < D_ACT_THRESHOLD


def test_bandwidth_constant_sane():
    assert 0.0 < SIGMA_BW <= 1.0