# MMD-SPRT Verification Harness

Distributional-attestation + SPRT verification pipeline gating deployment of a guarded subsystem.
The harness gates deployment by verifying that incoming telemetry stays distributionally aligned
with production (**Gate 0**), that the SPRT exponential-decay convergence diagnostic passes
(**SPRT**), and that offline evaluation shows sufficient power and a strong IPW correlation
(**Phase 4**).

**Primary implementation:** Rust (`crates/`), **reference cross-check:** Python (`harness/`, `tests/`).

**Verified at HEAD:** Rust 46 tests, Python 64 tests, 0 flake8 issues, 0 Rust warnings.

## How to reproduce everything

### macOS / dev host (pure-Rust kernels + Python reference)

```bash
# Clone and enter the repo
git clone https://github.com/VasanthSaravanann/mmd-sprt-harness.git
cd mmd-sprt-harness

# Rust workspace (eBPF crates excluded from default build)
cargo test                                # 46 tests
cargo bench -p harness-pipeline           # 5 criterion benchmarks
cargo run --release -p harness-pipeline --bin bench_report  # → benches/criterion_summary.json

# Python reference suite
python3 -m venv .venv
.venv/bin/pip install pytest numpy scipy scikit-learn matplotlib psutil
.venv/bin/python -m pytest tests/ -q      # 64 tests

# Generate all benchmark figures (Tasks 1-6, 8)
.venv/bin/python benches/run_all_benches.sh
# → paper/figures/fig1..fig9 generated
```

### Linux build host — eBPF verifier (requires root)

```bash
# One-time: nightly toolchain + compiler
rustup toolchain install nightly --component rust-src
cargo install bpf-linker

# Build the kernel eBPF program
cargo +nightly build -p ebpf-verifier-ebpf --release \
  --target bpfel-unknown-none -Z build-std=core

# Build the userspace loader
cargo build -p ebpf-verifier --release

# Run the loader standalone (telemetry stream to stderr):
sudo ./target/release/ebpf-verifier \
  --path target/bpfel-unknown-none/release/ebpf-verifier-ebpf

# Full benchmark (SIGKILL latency + ringbuf throughput under load):
sudo .venv/bin/python benches/eval_ebpf_bench.py \
  --ebpf-obj target/bpfel-unknown-none/release/ebpf-verifier-ebpf \
  --output benches/ebpf-bench --stress
# → benches/ebpf-bench/plain/ and benches/ebpf-bench/stress/ directories with
#   container_performance.json, fig10_*.pdf, fig11_*.pdf
```

## Workspace layout

```
.
├── Cargo.toml                         workspace root
├── crates/
│   ├── harness-core/            46 tests — MMD², SPRT, IPW
│   ├── harness-pipeline/        e2e pipeline + criterion benches
│   ├── ebpf-verifier/           aya userspace loader (Linux root only)
│   └── ebpf-verifier-ebpf/      no_std kernel program (bpfel target)
├── harness/                      Python reference (gate0, sprt, evaluator, pipeline)
├── tests/                        64 Python tests
├── benches/                      Bench scripts + orchestrator
├── benches/ebpf-bench/           Real eBPF measurements (from Linux host)
│   ├── plain/                    Unloaded results (fig10, fig11, container_performance.json)
│   └── stress/                   Under stress-ng CPU load
└── paper/figures/                Generated figure PDFs (fig1-fig11)
```

## eBPF real benchmark results

Measured on an x86_64 Linux host with `benches/eval_ebpf_bench.py`. The script
forks a child that opens the canary file `/tmp/canary_token.key`; the kernel
program fires `bpf_send_signal(SIGKILL)` from `__x64_sys_openat` (and now also
`__x64_sys_open` and `__x64_sys_openat2` to cover musl/Alpine and `openat2`).
Latency is the wall-clock time from `fork()` to `waitpid()` receiving SIGKILL.

### Plain (no added load)

| Metric | Value |
|---|---|
| SIGKILL latency median | 4.3 ms |
| SIGKILL latency p95 | 5.4 ms |
| SIGKILL latency p99 | 6.0 ms |
| SIGKILL latency max | 7.8 ms |
| Ringbuf throughput (8 threads) | 69,016 events/s |
| Loader CPU (mean / peak) | 0.6% / 5.0% |
| Loader RSS | 37.9 MB |

### Under stress-ng CPU load (all cores busy)

| Metric | Value |
|---|---|
| SIGKILL latency median | 4.2 ms |
| SIGKILL latency p95 | 5.7 ms |
| SIGKILL latency max | 6.9 ms |
| Loader CPU (mean / peak) | 0.9% / 10.0% |
| Loader RSS | 37.9 MB |

Note: the tail widens under CPU load (p95 goes from 5.4 ms to 5.7 ms), driven by
scheduler jitter delaying the child's timeslice.

## Pipeline stages

| Stage | Rust kernel | Verdict tokens |
|---|---|---|
| **Gate 0** – distributional attestation | `gate0.rs` | `GATE0_PASS` / `GATE0_FAIL` |
| **SPRT** – convergence diagnostic | `sprt.rs` | `PASS` / `INSUFFICIENT_FIT_HALT` / `BUDGET_EXHAUSTED_CONVERGING` / `NEEDS_MORE_DATA` |
| **Phase 4** – IPW correlation gate | `evaluator.rs` | `PASS` / `INSUFFICIENT_POWER` / `INSUFFICIENT_CORRELATION` / `NOT_SIGNIFICANT` |
| **Pipeline** – aggregate verdict | `run_pipeline` | `FULL_DEPLOYMENT_GRANTED` / `DEPLOYMENT_HALTED` |

## Benchmark figures

| Figure | Script | What it shows | Reproducible on |
|---|---|---|---|
| `fig1_gate0_sensitivity.pdf` | `benches/eval_gate0_sensitivity.py` | MMD² vs cosine across 0-5% injection rate | Any OS |
| `fig2_sprt_token_savings.pdf` | `benches/eval_sprt_efficiency.py` | Fixed-budget vs SPRT mean rounds (36% savings) | Any OS |
| `fig3_ipw_winsorization.pdf` | `benches/eval_ipw_variance.py` | P99 winsorization preserves n_eff≥30 | Any OS |
| `fig4_sprt_window_ablation.pdf` | `benches/eval_sprt_ablation.py` | Real exponential-fit false-decision rates | Any OS |
| `fig5_ebpf_latency.pdf` | `benches/eval_ebpf_race_window.py` | Simulated userspace vs eBPF (macOS) | macOS |
| `fig6_e2e_pipeline_matrix.pdf` | `benches/eval_e2e_joint_pipeline.py` | 0 false deployments, AUC=1.000 | Any OS |
| `fig7_agent_traces.pdf` | `benches/eval_agent_traces.py` | Suite A/B ROC curve (AUC=0.67) | Any OS |
| `fig8_baselines_roc.pdf` | `benches/eval_baselines.py` | Detector comparison + OPE estimators | Any OS |
| `fig9_baselines_ope.pdf` | `benches/eval_baselines.py` | OPE MAE + Spearman r | Any OS |
| `fig10_ebpf_enforcement_latency.pdf` | `benches/eval_ebpf_bench.py` | Real SIGKILL latency distribution | **Linux + root** |
| `fig11_ebpf_ringbuf_throughput.pdf` | `benches/eval_ebpf_bench.py` | Ringbuf throughput bar chart | **Linux + root** |

Figures 1-9 can be regenerated with `benches/run_all_benches.sh` (macOS/dev host).
Figures 10-11 require a Linux host with root: `benches/eval_ebpf_bench.py`.

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

- P99 winsorization via **lower** percentile rank `floor(p·(n−1))`. NaN-safe in both languages.
- Weighted **Spearman** (rank-based) correlation; exact two-sided Student-t p-value
  via the regularized incomplete beta — identical in Python and Rust.
- Production subsample is a **seeded uniform random 1% draw**.

| Condition | Status token |
|---|---|
| `n_eff < 30` | `INSUFFICIENT_POWER` |
| `r_{2,IPW} ≤ 0.85` | `INSUFFICIENT_CORRELATION` |
| `p ≥ 0.01` | `NOT_SIGNIFICANT` |
| All passed | `PASS` |

## Known limitations

- The `ebpf-verifier` and `ebpf-verifier-ebpf` crates target `bpfel-unknown-none` and
  require a Linux host + nightly toolchain + root; excluded from workspace `default-members`.
- Benchmark Task 7's ≥15× claim is measured against the **scalar Python reference kernel**
  this Rust port replaces; vectorized NumPy is faster at these matrix sizes
  (see `benches/criterion_summary.json`).