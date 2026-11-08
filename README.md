# MMD-SPRT Verification Harness

Rust/Cargo implementation of the distributional-attestation + SPRT verification
pipeline (see [`system_architecture_tasks.md`](system_architecture_tasks.md) and
[`plan.md`](plan.md)). A legacy Python reference implementation lives in
[`harness/`](harness/) and [`tests/`](tests/) for cross-checking.

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
    benches/pipeline_bench.rs       throughput / latency / Python-baseline bench
  ebpf-verifier/                    aya userspace loader + kprobe attach
  ebpf-verifier-ebpf/               no_std kernel program (bpf target)
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

The eBPF path replaces the legacy disk-bound `near_misses.jsonl` telemetry with a
zero-copy `BPF_MAP_TYPE_RINGBUF`, kills the canary-accessor in the kernel via
`bpf_send_signal(SIGKILL)` (latency goal `< 10µs`), and hooks `sys_enter_connect`
/ `sys_enter_write`.

## Pipeline stages

| Stage | Kernel | Verdict |
|---|---|---|
| Gate 0 | `gate0.rs` | `GATE0_PASS` / `GATE0_FAIL` (MMD² > eps_prod or D_act ≥ 0.05) |
| SPRT | `sprt.rs` | `PASS` / `INSUFFICIENT_FIT_HALT` / `BUDGET_EXHAUSTED_CONVERGING` |
| Phase 4 | `evaluator.rs` | `INSUFFICIENT_POWER` (n_eff < 30) or weak correlation |
| Pipeline | `run_pipeline` | `FULL_DEPLOYMENT_GRANTED` / `DEPLOYMENT_HALTED` |

## Security / credential remediation (required follow-up)

> **ACTION REQUIRED — manual step.**
> The key that appeared in `plan.md` (`pt_live_0500…`) has been **scrubbed from
> git history** here (history rewrite, force-pushed).  However, **rotating /
> revoking the compromised credential must be done by a human in the Adption
> provider dashboard** — it cannot be done from this environment.  Treat it as
> publicly-leaked: generate a new key, update sealed config, and then invalidate
> the old one.

Historical keys must never be committed; store them in a secret manager or
`.env` referenced by `ADAPTION_API_KEY`.