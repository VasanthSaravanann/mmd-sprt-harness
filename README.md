# MMD-SPRT Verification Harness

Distributional-attestation + SPRT verification pipeline gating deployment of a guarded subsystem.
Implements four gates — Gate 0 distributional alignment, SPRT convergence, Phase 4 IPW correlation,
and a userspace verifier — in **Rust** (primary) and **Python** (reference/cross-check).

**Verified at commit `HEAD`:** Rust 46 tests, Python 64 tests, 0 flake8 issues, 0 Rust warnings.

## Workspace layout

```
.
├── Cargo.toml                          workspace (default members: harness-core, harness-pipeline)
├── crates/
│   ├── harness-core/                   MMD², SPRT, IPW kernels (pure Rust)
│   │   ├── src/
│   │   │   ├── gate0.rs                length-normalized 3-gram MMD² + Action Jaccard distance
│   │   │   ├── sprt.rs                 Parallel fuzzer + N ≥ 5 exponential-decay fit
│   │   │   ├── evaluator.rs            P99-winsorized weighted-Spearman IPW gate
│   │   │   └── lib.rs
│   │   └── tests/                      46 unit + integration tests
│   ├── harness-pipeline/              Unified pipeline wireup + criterion benches
│   │   ├── src/
│   │   │   ├── lib.rs                 run_pipeline() → verdict
│   │   │   └── bin/bench_report.rs    Native throughput report → JSON
│   │   ├── tests/                     e2e pipeline tests
│   │   └── benches/                   6 micro-benchmarks (spectrum, MMD², ringbuf, IPW)
│   ├── ebpf-verifier/                 aya userspace loader (Linux bpf build host)
│   └── ebpf-verifier-ebpf/            no_std kernel program (bpfel-unknown-none)
├── harness/                           Python reference implementation
├── tests/                             64 Python tests cross-checking the Rust kernels
├── benches/                           7 Python bench scripts + 1 orchestrator
│   └── run_all_benches.sh             Full pipeline: sweep → criterion → figure verification
└── paper/figures/                     6 generated PDFs (Tasks 1–6)
```

## Build & test

```bash
# Pure-Rust crates (macOS/dev hosts — eBPF crates are excluded from default build)
cargo test                              # 46 tests
cargo bench -p harness-pipeline         # 6 criterion benchmarks
cargo run --release -p harness-pipeline --bin bench_report  # → benches/criterion_summary.json

# Python reference suite (cross-check)
cd /path/to/mmd-sprt-harness
python3 -m venv .venv
.venv/bin/pip install pytest numpy scipy scikit-learn matplotlib
.venv/bin/python -m pytest tests/ -q   # 64 tests
.venv/bin/python benches/run_all_benches.sh  # Full benchmark sweep + figure generation
```

### eBPF program (requires separate Linux build host)

```bash
rustup target add bpfel-unknown-none
cargo build -p ebpf-verifier-ebpf --target bpfel-unknown-none
sudo cargo run -p ebpf-verifier
```

## Pipeline stages

| Stage | Kernel (Rust) | Verdict tokens |
|---|---|---|
| **Gate 0** – distributional attestation | `gate0.rs` | `GATE0_PASS` / `GATE0_FAIL` |
| **SPRT** – convergence diagnostic | `sprt.rs` | `PASS` / `INSUFFICIENT_FIT_HALT` / `BUDGET_EXHAUSTED_CONVERGING` / `NEEDS_MORE_DATA` |
| **Phase 4** – IPW correlation gate | `evaluator.rs` | `PASS` / `INSUFFICIENT_POWER` / `INSUFFICIENT_CORRELATION` / `NOT_SIGNIFICANT` |
| **Pipeline** – aggregate verdict | `run_pipeline` | `FULL_DEPLOYMENT_GRANTED` / `DEPLOYMENT_HALTED` |

Each stage exposes a canonical `status: &str` field (e.g. `GATE0_FAIL`, `INSUFFICIENT_CORRELATION`) in both the Rust struct and the Python dataclass, so the pipeline table contract is uniform across implementations.

## Mathematical notes

### Gate 0 — 3-gram MMD² + Action Jaccard

Length-normalized 3-gram spectra via a Gaussian-RBF kernel (bandwidth `σ = 0.15`).  
Production floor: `ε_prod = max(MMD²_prod‑prod, 0.015)`.  
The gate asserts `MMD²(cand, prod) ≤ ε_prod` **and** `D_act < 0.05`.

A 1% *tail-exfiltration* perturbation (withheld tokens injected into the candidate)
raises MMD² above `ε_prod` while leaving the mean-embedding cosine distance flat
(~0.0007) — the exact attack scenario that motivated the MMD² floor.

### SPRT — exponential-decay fit

Fits `d^(i) = β₀·e^(−λi)` over a rolling `N ≥ 5` window (zero-based indices).
Passes iff `R² ≥ 0.90` **and** extrapolated `d⁽⁶⁾ < 0.02`; `R² < 0.90` → `INSUFFICIENT_FIT_HALT`.
A good fit with `d⁽⁶⁾ ≥ 0.02` after exhausting the budget → `BUDGET_EXHAUSTED_CONVERGING`.
Fewer than 5 points → `NEEDS_MORE_DATA` (pre-fit guard, not a panic).

### Phase 4 — IPW weighted-Spearman correlation

- Inverse-propensity weights winsorized at P99 using the **lower** percentile rank
  `floor(p·(n−1))` — the previous `ceil` silently left a single outlier uncapped.
- NaN-safe: NaN values are excluded from the percentile rank and preserved in the output.
- Weighted **Spearman** (rank-based) correlation (both Python and Rust; an earlier
  Python implementation used raw-value Pearson).
- Exact two-sided Student-t p-value via the regularized incomplete beta
  `I_{ν/(ν+t²)}(ν/2, ½)` — identical in both languages, verified against `scipy.stats.t`
  to `< 1e-12` (Python) / `< 1e-7` (Rust Lanczos `ln_gamma`).
- Production subsample is a **seeded uniform random 1% draw** (deterministic first-N
  was fixed to match the spec's "randomized" requirement).

| Condition | Status token |
|---|---|
| `n_eff < 30` | `INSUFFICIENT_POWER` |
| `r_{2,IPW} ≤ 0.85` | `INSUFFICIENT_CORRELATION` |
| `p ≥ 0.01` | `NOT_SIGNIFICANT` |
| All passed | `PASS` |

## Benchmark figures (Tasks 1–8)

| Figure | Script | Assertion |
|---|---|---|
| `fig1_gate0_sensitivity.pdf` | `benches/eval_gate0_sensitivity.py` | Cosine flat (~0.0007); 3-gram MMD² crosses 0.015 at 1% injection |
| `fig2_sprt_token_savings.pdf` | `benches/eval_sprt_efficiency.py` | ≥ 35% token savings, FPR < 5% (1000 rollouts) |
| `fig3_ipw_winsorization.pdf` | `benches/eval_ipw_variance.py` | Raw collapses `n_eff < 30`; P99 preserves `n_eff ≥ 30` up to α = 1.2 |
| `fig4_sprt_window_ablation.pdf` | `benches/eval_sprt_ablation.py` | N=3 false-decision 82%; N=5 drops to 4.9% with 38% savings |
| `fig5_ebpf_latency.pdf` | `benches/eval_ebpf_race_window.py` | Userspace ~2.9ms; eBPF < 10µs (simulated on macOS) |
| `fig6_e2e_pipeline_matrix.pdf` | `benches/eval_e2e_joint_pipeline.py` | 0 false deployments, ROC-AUC 1.000 (5000 trajectories) |

All figures generated by `benches/run_all_benches.sh` and stored in `paper/figures/`.
Rust MMD² throughput is **~37×** the scalar Python reference; a vectorized NumPy baseline
is reported separately in `benches/criterion_summary.json`.

## Known limitations

- The `ebpf-verifier` and `ebpf-verifier-ebpf` crates target `bpfel-unknown-none` and
  require a Linux host + nightly toolchain; they are excluded from the workspace
  `default-members`.
- The eBPF latency benchmark (Task 5) is **simulated** on macOS (no BPF VM available).
- Benchmark Task 7's ≥15× claim is measured against the **scalar Python reference kernel**
  this Rust port replaces; the vectorized NumPy baseline is faster at these small matrix
  sizes and is reported separately.
