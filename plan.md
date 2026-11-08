#Plan
## Environment Setup
```bash
export ADAPTION_API_KEY="${ADAPTION_API_KEY:-load from .env — never commit}"
mkdir -p harness tests
pip install pytest numpy scipy scikit-learn
```
---
## Task 1: Userspace Verifier Abstraction Proof
- **Target Files:** `harness/mock_verifier.py`, `tests/test_mock_verifier.py`
- **Spec:** Mock verifier on UNIX socket `/tmp/verifier.sock`. Validate telemetry schema (`near_misses.jsonl`), cascade Tier 1/2 events without model judges, mock `<2ms` SIGKILL routing.

Create harness/mock_verifier.py and tests/test_mock_verifier.py for Task 1. Listen on socket /tmp/verifier.sock, validate telemetry schema against near_misses.jsonl, cascade Tier 1/2 events without model judges, mock <2ms SIGKILL mechanics."
pytest tests/test_mock_verifier.py -v

---
## Task 2: Distributional Attestation Engine (Gate 0)
- **Target Files:** `harness/gate0.py`, `tests/test_gate0.py`
- **Spec:** Length-normalized 3-gram MMD² with floor $\epsilon_{\text{prod}} = \max(\text{MMD}^2_{\text{prod-prod}}, 0.015)$ and Action Jaccard $D_{\text{act}} < 0.05$. Fail when mean-embedding cosine distance passes 1% tail exfiltration tokens.

Create harness/gate0.py and tests/test_gate0.py for Task 2. Implement length-normalized 3-gram MMD² with eps_prod = max(MMD2_prod_prod, 0.015) and Action Jaccard D_act < 0.05. Test that Gate 0 fails on 1% tail exfiltration tokens where cosine distance passes."
pytest tests/test_gate0.py -v
---
## Task 3: Prior-Anchored SPRT Iteration Loop
- **Target Files:** `harness/fuzzer_sprt_loop.py`, `tests/test_fuzzer_sprt.py`
- **Spec:** Fixed-point loop ($k_{\text{max}} = 6$), stream mutation ($K \in [3, 8]$), Wilson 95% CI viability $V_E$, logit entropy $H_{\text{fuzz}}$. Fit exponential decay $d^{(i)} = \beta_0 e^{-\lambda i}$ on $N \ge 5$ rolling window ($\{d^{(1)}, \dots, d^{(5)}\}$). PASS iff $R^2 \ge 0.90$ AND $d^{(6)} < 0.02$. Trigger `INSUFFICIENT_FIT_HALT` if $R^2 < 0.90$.

Create harness/fuzzer_sprt_loop.py and tests/test_fuzzer_sprt.py for Task 3. Fixed-point loop k_max=6, mutated streams K in [3,8], Wilson 95% CI V_E, logit entropy H_fuzz. Fit exponential decay d^{(i)} = beta_0 * exp(-lambda * i) over rolling window N >= 5 points. PASS iff R^2 >= 0.90 and d^{(6)} < 0.02; trigger INSUFFICIENT_FIT_HALT if R^2 < 0.90. Verify [0.30, 0.20, 0.12, 0.07, 0.04] passes and [0.30, 0.10, 0.28, 0.05, 0.22] halts."
pytest tests/test_fuzzer_sprt.py -v

---
## Task 4: IPW Correlation Evaluator (Phase 4 Gate)
- **Target Files:** `harness/evaluator.py`, `tests/test_evaluator.py`
- **Spec:** Combine 1% randomized production data with stratified high-$V_E$ evaluation data. Winsorize weights at $P_{99}$ to enforce $n_{\text{eff}} \ge 30$. PASS iff $n_{\text{eff}} \ge 30$ and $r_{2,\text{IPW}} > 0.85$ ($p < 0.01$).

Create harness/evaluator.py and tests/test_evaluator.py for Task 4. Combine 1% production data with stratified high-V_E evaluation data. Winsorize IPW weights at P_99, enforce n_eff >= 30, pass iff n_eff >= 30 and r_{2,IPW} > 0.85 (p < 0.01)."
pytest tests/test_evaluator.py -v

---
## Task 5: Master Pipeline Integration Test
- **Target File:** `tests/test_e2e_pipeline.py`
- **Spec:** Connect Gate 0 through Phase 4 in unified harness. Run across baseline flawed system suite (assert failure) and updated robust system suite (assert PASS).

Create tests/test_e2e_pipeline.py connecting Gate 0 through Phase 4. Test execution across flawed baseline system (assert failure) and updated robust system (assert PASS).
pytest tests/test_e2e_pipeline.py -v

## Architectural Tradeoffs & System Limitations

| Metric / Component | Primary Advantage | Tradeoff / Limitation |
| :--- | :--- | :--- |
| **$N \ge 5$ Decay Window** | Eliminates false halting caused by single mini-batch noise ($df=3$). | Increases cold-start sampling requirements before making early-stopping calls ($k_{\text{max}}=6$). |
| **3-gram $\text{MMD}^2$ Floor** | Detects subtle 1% tail exfiltration attacks missed by cosine distance. | Has $O(n^2)$ sequence length complexity compared to linear $O(n)$ mean-embeddings. |
| **IPW $P_{99}$ Winsorization** | Prevents variance explosion and guarantees $n_{\text{eff}} \ge 30$. | Introduces slight statistical bias under extreme distribution shifts. |
| **UNIX Socket Verifier** | Validates telemetry routing deterministically prior to kernel eBPF driver integration. | Serves strictly as an abstraction proof; does not measure production kernel-level mitigation latency. |
---
## Full Verification

pytest tests/ -v --tb=short
