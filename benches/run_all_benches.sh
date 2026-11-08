#!/usr/bin/env bash
# Task 8 — Master evaluation execution command & plot artifact aggregation.
#
# 1. Create paper/figures/.
# 2. Execute the parameter-sweep scripts (Tasks 1-6) sequentially.
# 3. Run the Cargo Criterion benchmark suite (Task 7).
# 4. Verify all vector plot PDFs (fig1..fig6) exist, are non-empty, and are
#    ready for LaTeX inclusion.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./.venv/bin/python
FIG=paper/figures
mkdir -p "$FIG"

echo "== Task 1: Gate 0 MMD^2 sensitivity sweep =="
$PY benches/eval_gate0_sensitivity.py
echo "== Task 2: SPRT efficiency =="
$PY benches/eval_sprt_efficiency.py
echo "== Task 3: IPW variance / winsorization =="
$PY benches/eval_ipw_variance.py
echo "== Task 4: SPRT window ablation =="
$PY benches/eval_sprt_ablation.py
echo "== Task 5: eBPF race-window benchmark (simulated) =="
$PY benches/eval_ebpf_race_window.py
echo "== Task 6: E2E joint pipeline matrix =="
$PY benches/eval_e2e_joint_pipeline.py

echo "== Task 7: Rust Criterion bench + native report =="
cargo bench -p harness-pipeline
cargo run --release -p harness-pipeline --bin bench_report

echo "== Baselines + Rust MMD^2 speedup =="
$PY benches/_numpy_mmd2_baseline.py
$PY - <<'EOF'
import json, os, sys, time
sys.path.insert(0, os.getcwd())
import random
import harness.gate0 as g

# scalar Python REFERENCE kernel (the baseline this Rust port replaces)
rng = random.Random(0)
MOTIFS=[['a','b','b','a'],['a','a','b','a'],['b','a','a','b']]
S = [ [rng.choice(MOTIFS)[i%4] for i in range(64)] for _ in range(48) ]
g.mmd2_3gram(S, S, 0.10)
t0 = time.perf_counter()
for _ in range(8):
    g.mmd2_3gram(S, S, 0.10)
python_calls = 1.0 / ((time.perf_counter() - t0) / 8)

with open("benches/criterion_summary.json") as f:
    rust = json.load(f)
with open("benches/numpy_baseline.json") as f:
    npb = json.load(f)
rust_calls = rust["mmd2_per_sec"]
ratio = rust_calls / python_calls
rust["rust_vs_python_ref_ratio"] = ratio
rust["python_ref_mmd2_calls_per_sec"] = python_calls
rust["numpy_mmd2_calls_per_sec"] = npb["numpy_mmd2_calls_per_sec"]
with open("benches/criterion_summary.json", "w") as f:
    json.dump(rust, f, indent=2)
print(f"[Task7] Rust MMD2 {rust_calls:.1f} calls/s vs Python-ref {python_calls:.1f} calls/s -> {ratio:.1f}x")
print(f"[Task7] (informational) NumPy vectorized baseline {npb['numpy_mmd2_calls_per_sec']:.1f} calls/s")
assert ratio >= 15, f"SIMD Rust MMD^2 must be >=15x the Python reference, got {ratio:.1f}x"
print("[Task7] OK: Rust >=15x Python-reference throughput")
EOF

echo "== Verify figure artifacts =="
missing=0
for i in 1 2 3 4 5 6; do
    f="$FIG/fig${i}_"*.pdf
    : ${f} # expand
    file=$(ls "$FIG"/fig${i}_*.pdf 2>/dev/null | head -1)
    if [ -n "$file" ] && [ -s "$file" ]; then
        echo "  OK  $file ($(wc -c < "$file") bytes)"
    else
        echo "  MISSING  fig${i}.pdf"
        missing=$((missing+1))
    fi
done
if [ "$missing" -ne 0 ]; then
    echo "ERROR: $missing figure(s) missing or empty" >&2
    exit 1
fi
echo "All 6 figures present and non-empty. Benchmarks complete."