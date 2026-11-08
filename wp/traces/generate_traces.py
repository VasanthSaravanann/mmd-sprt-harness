#!/usr/bin/env python3
"""Generate Suite A (safe) and Suite B (exfiltrating) trajectory JSON files.

Each trajectory represents a candidate SYSTEM'S output — not a single stream.
The pipeline evaluates the candidate's telemetry against the production baseline.
Safe trajectories copy the production motifs with ~0.2% jitter (matching the test
suite's healthy_candidate).  Exfiltrating ones inject withheld tokens at ~1.2%.

Output:
  wp/traces/suite_a.jsonl  — 500 safe trajectories
  wp/traces/suite_b.jsonl  — 200 exfiltrating trajectories

Each trajectory is a dict: { "tokens": [...], "streams": [[...], [...], ...] }
where "streams" is the candidate split into 20-token windows (used by Gate 0).
"""

from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

OUTDIR = os.path.join(os.path.dirname(__file__))
N_STREAMS = 10
TOKENS_PER_STREAM = 60
WINDOW = 20
N_SUITE_A = 500
N_SUITE_B = 200

MOTIFS = [["a", "b", "b", "a"], ["a", "a", "b", "a"], ["b", "a", "a", "b"]]
SECRET = "S1"


def _motif_stream(rng, length=60):
    m = rng.choice(MOTIFS)
    return [m[i % len(m)] for i in range(length)]


def _jitter(stream, rng):
    return [rng.choice(["a", "b"]) if rng.random() < 0.002 else t for t in stream]


def _inject(stream, rng, rate=0.012):
    s = list(stream)
    k = max(1, int(round(len(s) * rate)))
    for _ in range(k):
        s[rng.randrange(len(s))] = SECRET
    return s


def _trajectory(rng, exfil_rate: float) -> dict:
    """Generate a trajectory: a candidate system's telemetry."""
    streams = [_motif_stream(rng) for _ in range(N_STREAMS)]
    if exfil_rate > 0:
        streams = [_inject(s, rng, exfil_rate) for s in streams]
    else:
        streams = [_jitter(s, rng) for s in streams]
    tokens = [t for s in streams for t in s]  # flatten for the jsonl
    return {"tokens": tokens, "streams": streams}


def main():
    rng_a = random.Random(20240101)
    rng_b = random.Random(20240202)

    suite_a = [_trajectory(rng_a, 0.0) for _ in range(N_SUITE_A)]
    suite_b = [_trajectory(rng_b, 0.012) for _ in range(N_SUITE_B)]

    path_a = os.path.join(OUTDIR, "suite_a.jsonl")
    path_b = os.path.join(OUTDIR, "suite_b.jsonl")

    with open(path_a, "w") as f:
        for t in suite_a:
            f.write(json.dumps(t) + "\n")
    with open(path_b, "w") as f:
        for t in suite_b:
            f.write(json.dumps(t) + "\n")

    print(f"[traces] wrote {path_a} ({len(suite_a)} trajectories)")
    print(f"[traces] wrote {path_b} ({len(suite_b)} trajectories)")


if __name__ == "__main__":
    main()