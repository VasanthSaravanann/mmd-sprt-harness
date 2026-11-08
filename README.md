# MMD-SPRT Verification Harness

Distributional-attestation + SPRT verification pipeline gating deployment of a guarded subsystem.
The harness gates deployment by verifying that incoming telemetry stays distributionally aligned
with production (**Gate 0**), that the adaptive fuzzing loop is converging (**SPRT**), and that
offline evaluation shows sufficient power and a strong IPW correlation (**Phase 4**).

**Primary implementation:** Rust (src in `crates/`), **reference cross-check:** Python (src in `harness/`).

**Verified at HEAD:** Rust 46 tests, Python 64 tests, 0 flake8 issues, 0 Rust warnings.

## Environment

### macOS / dev host (pure-Rust kernels + Python reference)

```bash
git clone https://github.com/VasanthSaravanann/mmd-sprt-harness.git
cd mmd-sprt-harness

# Rust workspace (eBPF crates excluded from default build)
cargo test                                # 46 tests
cargo bench -p harness-pipeline           # 6 criterion benchmarks

# Python reference suite
python3 -m venv .venv
.venv/bin/pip install pytest numpy scipy scikit-learn matplotlib psutil
.venv/bin/python -m pytest tests/ -q      # 64 tests

# Full benchmark sweep (Tasks 1–6, 8) — generates paper/figures/*.pdf
.venv/bin/python benches/run_all_benches.sh
```

### Linux build host (eBPF verifier — requires root)

```bash
# Nightly toolchain + bpf-linker (one time)
rustup toolchain install nightly --component rust-src
cargo install bpf-linker

# Build the kernel eBPF program
cargo +nightly build -p ebpf-verifier-ebpf --release \
  --target bpfel-unknown-none -Z build-std=core

# Build the userspace loader
cargo build -p ebpf-verifier --release

# Benchmarks (root required for eBPF)
sudo .venv/bin/python benches/eval_ebpf_bench.py \
  --ebpf-obj target/bpfel-unknown-none/release/ebpf-verifier-ebpf \
  --output benches/ebpf-bench --stress

# To run the loader standalone (telemetry stream to stderr):
sudo ./target/release/ebpf-verifier \
  --path target/bpfel-unknown-none/release/ebpf-verifier-ebpf
```

## Workspace layout

```
.
├── Cargo.toml                          workspace root
├── crates/
│   ├── harness-core/            46 tests
│   │   ├── src/gate0.rs         MMD² + Action Jaccard
│   │   ├── src/sprt.rs          Parallel fuzzer + N≥5 decay fit
│   │   └── src/evaluator.rs     P99-winsorized weighted-Spearman IPW gate
│   ├── harness-pipeline/        e2e pipeline + criterion benches
│   │   ├── src/lib.rs           run_pipeline() → verdict
│   │   └── src/bin/bench_report.rs  → benches/criterion_summary.json
│   ├── ebpf-verifier/           aya userspace loader (Linux only)
│   └── ebpf-verifier-ebpf/      no_std kernel program (bpfel target)
├── harness/                      Python reference (gate0, sprt, evaluator, mock_verifier, pipeline)
├── tests/                        64 Python tests
├── benches/                      Benchmark scripts
│   ├── run_all_benches.sh        macOS: Tasks 1–6, 8
│   ├── eval_ebpf_bench.py        Linux: real SIGKILL latency, ringbuf throughput, CPU/RSS
│   └── ebpf-bench/               Real eBPF results (generated on Linux)
└── paper/figures/                6 simulated PDFs (Tasks 1–6)
```

## Pipeline stages

| Stage | Rust kernel | Verdict tokens |
|---|---|---|
| **Gate 0** – distributional attestation | `gate0.rs` | `GATE0_PASS` / `GATE0_FAIL` |
| **SPRT** – convergence diagnostic | `sprt.rs` | `PASS` / `INSUFFICIENT_FIT_HALT` / `BUDGET_EXHAUSTED_CONVERGING` / `NEEDS_MORE_DATA` |
| **Phase 4** – IPW correlation gate | `evaluator.rs` | `PASS` / `INSUFFICIENT_POWER` / `INSUFFICIENT_CORRELATION` / `NOT_SIGNIFICANT` |
| **Pipeline** – aggregate verdict | `run_pipeline` | `FULL_DEPLOYMENT_GRANTED` / `DEPLOYMENT_HALTED` |

## eBPF real benchmark results

Measured on Linux (x86_64) by `benches/eval_ebpf_bench.py` — the script forks a child
that opens the canary file `/tmp/canary_token.key`; the kernel program fires
`bpf_send_signal(SIGKILL)` from the `__x64_sys_openat` kprobe. The parent times
the child's lifespan via `waitpid`.

| Metric | Value | Target |
|---|---|---|
| **SIGKILL latency median** | **4.5 µs** | < 10 µs ✅ |
| SIGKILL latency p95 | 4.8 µs | < 10 µs ✅ |
| SIGKILL latency p99 | 4.9 µs | < 10 µs ✅ |
| SIGKILL latency max | 6.7 µs | < 10 µs ✅ |
| **Ringbuf throughput** | **51,580 events/s** | — |
| **Loader CPU** | **1.2% mean** (10% peak) | — |
| **Loader RSS** | **37.7 MB** | — |

Full output at `benches/ebpf-bench/` (fig5_ebpf_real_latency.pdf, .png, .json).

## Benchmark figures

| Figure | Script | Assertion | Reproducible on |
|---|---|---|---|
| `fig1_gate0_sensitivity.pdf` | `benches/eval_gate0_sensitivity.py` | Cosine flat (~0.0007); MMD² crosses 0.015 at 1% injection | Any OS (no deps) |
| `fig2_sprt_token_savings.pdf` | `benches/eval_sprt_efficiency.py` | ≥ 35% savings, FPR < 5% | Any OS |
| `fig3_ipw_winsorization.pdf` | `benches/eval_ipw_variance.py` | Raw n_eff < 30; winsorized ≥ 30 to α=1.2 | Any OS |
| `fig4_sprt_window_ablation.pdf` | `benches/eval_sprt_ablation.py` | N=3 false-decision 82%; N=5 drops to 4.9% | Any OS |
| `fig5_ebpf_latency.pdf` | `benches/eval_ebpf_race_window.py` | Simulated userspace/eBPF distributions | macOS (sim) |
| `fig5_ebpf_real_latency.pdf` | `benches/eval_ebpf_bench.py` | Real eBPF SIGKILL latency 4.5 µs | **Linux + root** |
| `fig6_e2e_pipeline_matrix.pdf` | `benches/eval_e2e_joint_pipeline.py` | 0 false deployments, AUC 1.000 | Any OS |

Rust MMD² throughput is **~37×** the scalar Python reference; vectorized NumPy
baseline is faster at these matrix sizes (see `benches/criterion_summary.json`).

## Mathematical notes

### Gate 0 — 3-gram MMD² + Action Jaccard

Length-normalized 3-gram spectra via Gaussian-RBF kernel (bandwidth `σ = 0.15`).
Production floor: `ε_prod = max(MMD²_prod‑prod, 0.015)`. Gate asserts
`MMD²(cand, prod) ≤ ε_prod` **and** `D_act < 0.05`.

A 1% *tail-exfiltration* perturbation (withheld tokens injected into the candidate)
raises MMD² above `ε_prod` while leaving mean-embedding cosine distance flat (~0.0007).

### SPRT — exponential-decay fit

Fits `d^(i) = β₀·e^(−λi)` over a rolling `N ≥ 5` window (zero-based indices).
**PASS** iff `R² ≥ 0.90` **and** `d⁽⁶⁾ < 0.02`; `R² < 0.90` → `INSUFFICIENT_FIT_HALT`.
Good fit with `d⁽⁶⁾ ≥ 0.02` → `BUDGET_EXHAUSTED_CONVERGING`.
Fewer than 5 points → `NEEDS_MORE_DATA` (no panic).

### Phase 4 — IPW weighted-Spearman correlation

- P99 winsorization via **lower** percentile rank `floor(p·(n−1))` (previous `ceil`
  silently left outliers uncapped). NaN-safe in both languages.
- Weighted **Spearman** (rank-based) correlation; exact two-sided Student-t p-value
  via the regularized incomplete beta `I_{ν/(ν+t²)}(ν/2, ½)` — identical in Python
  and Rust, verified against `scipy.stats.t` (< 1e-12 / < 1e-7).
- Production subsample is a **seeded uniform random 1% draw** (fixed first-N
  was corrected to match the "randomized" spec requirement).

| Condition | Status token |
|---|---|
| `n_eff < 30` | `INSUFFICIENT_POWER` |
| `r_{2,IPW} ≤ 0.85` | `INSUFFICIENT_CORRELATION` |
| `p ≥ 0.01` | `NOT_SIGNIFICANT` |
| All passed | `PASS` |

## Known limitations

- The `ebpf-verifier` and `ebpf-verifier-ebpf` crates target `bpfel-unknown-none` and
  require a Linux host + nightly toolchain + root; they are excluded from the workspace
  `default-members` and cannot be compiled or tested on macOS.
- Benchmark Task 7's ≥15× claim is measured against the **scalar Python reference kernel**
  this Rust port replaces; the vectorized NumPy baseline is faster at these small matrix
  sizes and is reported separately in `benches/criterion_summary.json`.