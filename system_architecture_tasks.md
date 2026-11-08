# Engineering Tasks & Technical Deliverables

---

## Task 1: Environment & Security Cleanup
* **Goal:** Remediate credential exposure and establish cargo workspace foundation.
* **Deliverables:**
  * Scrub `ADAPTION_API_KEY` from git history using `git-filter-repo` or BFG Repo-Cleaner.
  * Rotate/revoke the compromised key in your API provider dashboard.
  * Initialize the top-level Cargo workspace layout (`Cargo.toml` with crates: `ebpf-verifier`, `harness-core`, `harness-pipeline`).

---

## Task 2: eBPF Kernel Probe & Microsecond Canary Interceptor
*(Replaces Task 1 Mock)*

* **Target Files:**
  * `crates/ebpf-verifier/src/main.rs`
  * `crates/ebpf-verifier-ebpf/src/main.rs`
* **Deliverables:**
  * **Kernel Hooks:** Attach `kprobe`/`tracepoint` programs to `sys_enter_connect` and `sys_enter_write` using `aya` or `libbpf-rs`.
  * **In-Kernel Tier 2 Honeypot:** Intercept direct file path access to honeypots (`/tmp/canary_token.key`) inside the kernel and call `bpf_send_signal(SIGKILL)` directly, cutting execution latency to $< 10\mu\text{s}$.
  * **Zero-Copy Telemetry Stream:** Emit near-miss events through an eBPF `BPF_MAP_TYPE_RINGBUF` to userspace, replacing disk-bound `near_misses.jsonl` writes.

---

## Task 3: SIMD-Accelerated Gate 0 Distributional Attestation Crate
* **Target Files:**
  * `crates/harness-core/src/gate0.rs`
  * `crates/harness-core/tests/test_gate0.rs`
* **Deliverables:**
  * Implement length-normalized 3-gram character spectrum extraction using SIMD vector instructions (`packed_simd` / AVX2).
  * Port Gaussian-RBF $\text{MMD}^2$ kernel calculation and action-sequence Jaccard distance ($D_{\text{act}}$) to Rust.
  * Verify that standard cosine distance fails on a 1% tail perturbation, while SIMD $\text{MMD}^2$ correctly flags $\text{MMD}^2 > \epsilon_{\text{prod}}$ to output `GATE0_FAIL`.

---

## Task 4: Parallel Tokio/Rayon Fuzzer & $N \ge 5$ SPRT Convergence Diagnostics
* **Target Files:**
  * `crates/harness-core/src/sprt.rs`
  * `crates/harness-core/tests/test_sprt.rs`
* **Deliverables:**
  * **Async Fuzzer Engine:** Build a multi-worker pipeline using `tokio` (for stream ingestion) and `rayon` (for parallel Base64/Hex/prompt mutation rollouts and Wilson 95% CI score calculations).
  * **Rolling SPRT Engine:** Implement $K \in [3, 8]$ mini-batch evaluations and exponential decay fitting ($d^{(i)} = \beta_0 e^{-\lambda i}$) over an $N \ge 5$ rolling window using `nalgebra` or `ndarray`.
  * Verify convergence gate ($R^2 \ge 0.90 \to \text{BUDGET\_EXHAUSTED\_CONVERGING}$) and halt gate ($R^2 < 0.90 \to \text{INSUFFICIENT\_FIT\_HALT}$).

---

## Task 5: Fast IPW Correlation Evaluator Crate
* **Target Files:**
  * `crates/harness-core/src/evaluator.rs`
  * `crates/harness-core/tests/test_evaluator.rs`
* **Deliverables:**
  * Implement sample stratification combining 1% randomized production data with high-$V_E$ evaluation data.
  * Winsorize weights at $P_{99}$ and compute effective sample size $n_{\text{eff}}$.
  * Calculate weighted Spearman rank correlation $r_{2,\text{IPW}}(V_E, Y_{\text{prod}})$.
  * Enforce statistical circuit breaker gates: fail datasets with $n_{\text{eff}} < 30$ (`INSUFFICIENT_POWER`) or $r_{2,\text{IPW}} \le 0.85$.

---

## Task 6: Master End-to-End Pipeline Harness & Benchmarking
* **Target Files:**
  * `crates/harness-pipeline/src/lib.rs`
  * `tests/test_e2e_pipeline.rs`
  * `benches/pipeline_bench.rs`
* **Deliverables:**
  * Wire eBPF ringbuffer ingestion, Gate 0 $\text{MMD}^2$, Tokio fuzzer workers, SPRT diagnostics, and IPW correlation into a unified pipeline library.
  * Create integration tests mapping out full `FULL_DEPLOYMENT_GRANTED` vs `DEPLOYMENT_HALTED` run states.
  * Add `criterion` benchmarks comparing telemetry throughput, memory allocations, and evaluation latency against the Python baseline.