# MMD-SPRT Verification Harness

Rust/Cargo implementation of a distributional-attestation + SPRT verification
pipeline. The harness gates deployment of a guarded subsystem by verifying that
incoming telemetry stays distributionally aligned with production (Gate 0),
that the adaptive fuzzing loop is converging (SPRT), and that offline
evaluation shows sufficient power and a strong IPW correlation (Phase 4).

A legacy Python reference implementation lives in [`harness/`](harness/) and
[`tests/`](tests/) and is kept for cross-checking the Rust kernels.

## Workspace layout

```
Cargo.toml                          workspace (default members: core, pipeline)
crates/
  harness-core/                     Gate 0 MMD², SPRT, IPW kernels (pure Rust)
    src/gate0.rs                    length-normalized 3-gram MMD² + Jaccard
    src/sprt.rs                     tokio/rayon fuzzer + N≥5 decay fit
    src/evaluator.rs                P99-winsorized weighted-Spearman IPW gate
    tests/                          unit tests for each kernel
  harness-pipeline/                 unified end-to-end pipeline + criterion bench
    src/lib.rs                      run_pipeline() -> DEPLOYMENT verdict
    tests/test_e2e_pipeline.rs      FULL_DEPLOYMENT_GRANTED / DEPLOYMENT_HALTED
    benches/pipeline_bench.rs       throughput / latency benchmarks
  ebpf-verifier/                    aya userspace loader + kprobe attach
  ebpf-verifier-ebpf/               no_std kernel program (bpf target)
  harness/                          legacy Python reference implementation
  tests/                            Python reference test suite
```

## Build & test

```bash
# pure-Rust crates (works on macOS/dev hosts) — the eBPF crates are excluded
cargo test                              # workspace default members
cargo check -p harness-pipeline --benches
cargo bench -p harness-pipeline         # criterion: Gate 0 + IPW latency

# eBPF program — requires a Linux x86_64 host, nightly toolchain, and the Bpf target
rustup target add bpfel-unknown-none
cargo build -p ebpf-verifier-ebpf --target bpfel-unknown-none
# then load it with the userspace loader (root): sudo cargo run -p ebpf-verifier
```

### Python reference suite (optional cross-check)

```bash
python3 -m venv .venv && .venv/bin/pip install pytest numpy scipy scikit-learn
.venv/bin/python -m pytest tests/ -q
```

## Pipeline stages

| Stage | Kernel | Verdict |
|---|---|---|
| Gate 0 | `gate0.rs` | `GATE0_PASS` / `GATE0_FAIL` (MMD² > eps_prod or D_act ≥ 0.05) |
| SPRT | `sprt.rs` | `PASS` / `INSUFFICIENT_FIT_HALT` / `BUDGET_EXHAUSTED_CONVERGING` |
| Phase 4 | `evaluator.rs` | `INSUFFICIENT_POWER` (n_eff < 30) or weak correlation |
| Pipeline | `run_pipeline` | `FULL_DEPLOYMENT_GRANTED` / `DEPLOYMENT_HALTED` |

The eBPF path replaces the legacy disk-bound `near_misses.jsonl` telemetry with
a zero-copy `BPF_MAP_TYPE_RINGBUF`, kills the canary-accessor in the kernel via
`bpf_send_signal(SIGKILL)` (latency goal `< 10µs`), and hooks
`sys_enter_connect` / `sys_enter_write`.

## Mathematical notes

- **Gate 0** — length-normalized 3-gram spectra with a Gaussian-RBF kernel.
  The production floor is `eps_prod = max(MMD²_prod-prod, 0.015)`; the gate fails
  if `MMD² > eps_prod` or `D_act ≥ 0.05`. A 1% *tail-exfiltration* perturbation
  slips past a mean-embedding cosine detector while the 3-gram MMD² correctly
  flags it.
- **SPRT** — fits `d^(i) = β₀·e^(−λi)` over a rolling `N ≥ 5` window (zero-based
  indices). Passes iff `R² ≥ 0.90` **and** extrapolated `d⁽⁶⁾ < 0.02`; `R² < 0.90`
  halts with `INSUFFICIENT_FIT_HALT`.
- **Phase 4** — IPW weights are winsorized at P99 before computing the effective
  sample size `n_eff = (Σw)²/Σw²` and a weighted **Spearman** correlation
  `r_{2,IPW}`. Requires `n_eff ≥ 30` and `r_{2,IPW} > 0.85`.

## Security / credentials

The `ADAPTION_API_KEY` value was once committed in planning docs and is treated
as **publicly leaked**.

- **History** has been rewritten with `git-filter-repo` and force-pushed to
  purge the secret from all commits.
- The current live value is stored only in `.env` (**git-ignored**); a placeholder
  `.env.example` is tracked so the variable name stays documented.
- ⚠️ **Action still recommended:** rotate / revoke the leaked credential in the
  Adption provider dashboard (a manual step that cannot be done from this
  environment), then invalidate the old value.

Never commit real credentials — use a secret manager or `.env` that is
git-ignored and referenced as `${ADAPTION_API_KEY}`.