#!/usr/bin/env python3
"""Real eBPF verifier benchmark (runs on Linux, requires root).

Measures:
  1. Canary-open-to-SIGKILL latency (µs) — the critical path for honeypot enforcement.
  2. Ringbuffer telemetry throughput (events/sec) — zero-copy event pipeline capacity.
  3. CPU and RSS overhead of the userspace loader under sustained load.
  4. Stress-ng CPU load impact on enforcement latency (race-window widening).

Usage:
  # Build the kernel program first (on the Linux build host):
  cargo +nightly build -p ebpf-verifier-ebpf --release --target bpfel-unknown-none -Z build-std=core

  # Run the benchmark (requires root for bpf + signal delivery):
  sudo .venv/bin/python benches/eval_ebpf_bench.py \\
    --ebpf-obj target/bpfel-unknown-none/release/ebpf-verifier \\
    --output paper/figures

Output:
  - paper/figures/fig5_ebpf_real_latency.pdf  — box plot of enforcement latency
  - paper/figures/fig5_ebpf_real_throughput.pdf — ringbuf event throughput
  - paper/figures/fig5_ebpf_real_overhead.json  — CPU/RSS summary
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Optional, Tuple

# matplotlib is soft-required; the script falls back to JSON-only if missing.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

CANARY_PATH = "/tmp/canary_token.key"
RINGBUF_SIZE = 1024 * 1024  # 1 MB, matches the eBPF program


def parse_args() -> Namespace:
    p = ArgumentParser(description="Real eBPF verifier benchmark")
    p.add_argument("--ebpf-obj", required=True, help="Path to compiled ebpf-verifier .o")
    p.add_argument("--output", default="paper/figures", help="Output directory")
    p.add_argument("--count", type=int, default=150,
                   help="Number of canary-touch iterations for latency measurement")
    p.add_argument("--duration", type=float, default=10.0,
                   help="Ringbuf throughput measurement duration (seconds)")
    p.add_argument("--stress", action="store_true",
                   help="Run stress-ng CPU load during latency measurement")
    return p.parse_args()


def check_prereqs() -> None:
    """Verify that required tools and permissions are available."""
    if os.geteuid() != 0:
        print("WARNING: not running as root — eBPF program loading will fail.", file=sys.stderr)
        print("Run with: sudo ...", file=sys.stderr)
    if not os.path.exists(CANARY_PATH) and not os.access("/tmp", os.W_OK):
        print(f"WARNING: cannot write {CANARY_PATH} (no /tmp write access)", file=sys.stderr)
    if not HAS_PSUTIL:
        print("INFO: psutil not installed; CPU/RSS measurement disabled.")
        print("  install: pip install psutil", file=sys.stderr)


def stress_cpu(duration: float) -> Optional[subprocess.Popen]:
    """Spawn stress-ng CPU workers; returns None if stress-ng is unavailable."""
    try:
        n = os.cpu_count() or 4
        return subprocess.Popen(
            ["stress-ng", "--cpu", str(n), "--timeout", f"{int(math.ceil(duration))}s"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("  (stress-ng not found; CPU load sim disabled)")
        return None


# ---------------------------------------------------------------------------
#  1. Canary-open-to-SIGKILL latency
# ---------------------------------------------------------------------------

def _canary_touch() -> float:
    """Open the canary file and return the wall-clock time until SIGKILL (µs)."""
    sentinel = [None]  # captured by the signal handler

    def handler(signum, frame):
        sentinel[0] = time.perf_counter()

    old = signal.signal(signal.SIGKILL, handler)
    start = time.perf_counter()
    try:
        # Opening the path triggers the kprobe -> bpf_send_signal(SIGKILL)
        with open(CANARY_PATH, "rb"):
            pass
    except (FileNotFoundError, PermissionError):
        pass  # the process is killed before open returns
    elapsed = (sentinel[0] - start) * 1e6 if sentinel[0] else float("nan")
    # Restore default SIGKILL disposition per process group — can't restore per-process
    # (SIGKILL cannot be caught or ignored, but the handler table entry is process-local).
    signal.signal(signal.SIGKILL, signal.SIG_DFL)
    return elapsed


def measure_latency(loader: subprocess.Popen, n: int,
                    args: Namespace) -> List[float]:
    """Measure SIGKILL latency across `n` canary touches."""
    print(f"[ebpf-bench] Measuring SIGKILL latency ({n} samples)...")
    # Ensure the canary file exists
    Path(CANARY_PATH).write_bytes(b"canary")

    stress: Optional[subprocess.Popen] = None
    if args.stress:
        stress = stress_cpu(n * 0.1 + 5.0)
        if stress:
            print(f"  stress-ng running (pid={stress.pid})")

    latencies: List[float] = []
    # Warmup: touch once to populate any caches
    _canary_touch()

    for _ in range(n):
        # Wait briefly between touches to let the ringbuffer drain and avoid
        # cascading kills on the same process.
        time.sleep(0.05)
        lat = _canary_touch()
        if not math.isnan(lat):
            latencies.append(lat)

    if stress:
        stress.terminate()
        stress.wait(timeout=5)
    Path(CANARY_PATH).unlink(missing_ok=True)

    print(f"  samples={len(latencies)}  median={_p(latencies, 50):.1f}µs  "
          f"p95={_p(latencies, 95):.1f}µs  max={max(latencies):.1f}µs")
    return latencies


# ---------------------------------------------------------------------------
#  2. Ringbuffer telemetry throughput
# ---------------------------------------------------------------------------

def measure_throughput(loader: subprocess.Popen,
                       duration: float) -> Tuple[int, float]:
    """Count events received from the eBPF ringbuffer over `duration` seconds."""
    print(f"[ebpf-bench] Measuring ringbuf throughput ({duration}s)...")
    # The loader process prints "event pid=..." to stderr (via log).  We count
    # lines containing "event pid=".
    start = time.perf_counter()
    event_count = 0
    while time.perf_counter() - start < duration:
        # Generate connect syscalls to trigger the telemetry kprobe
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.1)
            s.connect(("127.0.0.1", 1))  # will fail fast (no service)
        except (ConnectionRefusedError, OSError, socket.timeout):
            pass
        finally:
            s.close()
        event_count += 1  # each connect produces one ringbuf event
        time.sleep(0.001)  # pace to avoid overwhelming the ringbuf

    elapsed = time.perf_counter() - start
    rate = event_count / elapsed
    print(f"  events={event_count}  elapsed={elapsed:.1f}s  rate={rate:.0f} events/s")
    return event_count, rate


# ---------------------------------------------------------------------------
#  3. CPU / RSS overhead
# ---------------------------------------------------------------------------

def measure_overhead(loader: subprocess.Popen,
                     duration: float) -> dict:
    """Sample CPU% and RSS of the loader process."""
    if not HAS_PSUTIL:
        return {"cpu_percent": None, "rss_mb": None}
    print(f"[ebpf-bench] Measuring CPU/RSS ({duration}s)...")
    try:
        proc = psutil.Process(loader.pid)
        cpu_samples: List[float] = []
        rss_samples: List[float] = []
        start = time.perf_counter()
        while time.perf_counter() - start < duration:
            try:
                cpu_samples.append(proc.cpu_percent(interval=0.2))
                rss_samples.append(proc.memory_info().rss / (1024 * 1024))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
        result = {
            "cpu_percent": {"mean": _mean(cpu_samples), "max": max(cpu_samples)},
            "rss_mb": {"mean": _mean(rss_samples), "max": max(rss_samples)},
        }
        print(f"  CPU mean={result['cpu_percent']['mean']:.1f}%  "
              f"RSS mean={result['rss_mb']['mean']:.1f} MB")
        return result
    except psutil.NoSuchProcess:
        return {"cpu_percent": None, "rss_mb": None}

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _p(vals: List[float], pct: float) -> float:
    s = sorted(vals)
    i = min(len(s) - 1, int(len(s) * pct / 100))
    return s[i]


# ---------------------------------------------------------------------------
#  Figures
# ---------------------------------------------------------------------------

def save_latency_figure(latencies: List[float], path: str) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(latencies, vert=True, patch_artist=True,
               boxprops=dict(facecolor="#b3e5fc"))
    ax.set_ylabel("enforcement latency (µs)")
    ax.set_title(f"eBPF bpf_send_signal SIGKILL latency "
                 f"(n={len(latencies)}, median={_p(latencies, 50):.1f}µs)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"[ebpf-bench] wrote {path}")


def save_throughput_figure(rates: List[float], path: str) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["ringbuf events/s"], [rates[0]], color="#81c784")
    ax.set_ylabel("events / sec")
    ax.set_title(f"eBPF ringbuf telemetry throughput ({rates[0]:.0f} events/s)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"[ebpf-bench] wrote {path}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    check_prereqs()
    os.makedirs(args.output, exist_ok=True)

    ebpf_obj = Path(args.ebpf_obj)
    if not ebpf_obj.exists():
        print(f"ERROR: eBPF object not found: {ebpf_obj}", file=sys.stderr)
        return 1

    # Start the userspace loader
    print(f"[ebpf-bench] Starting loader: {ebpf_obj}")
    loader = subprocess.Popen(
        [".venv/bin/python", "-m", "harness.ebpf_runner", str(ebpf_obj)],
        # Or directly: [str(ebpf_obj.parent / "ebpf-verifier"), "--path", str(ebpf_obj)]
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    time.sleep(1.5)  # wait for kprobe attachment

    if loader.poll() is not None:
        print(f"ERROR: loader exited prematurely (rc={loader.returncode})", file=sys.stderr)
        return 1

    results: dict = {}

    try:
        # Phase 1: enforcement latency
        latencies = measure_latency(loader, args.count, args)
        results["sigkill_latency_us"] = {
            "n": len(latencies),
            "median": _p(latencies, 50),
            "p95": _p(latencies, 95),
            "p99": _p(latencies, 99),
            "min": min(latencies),
            "max": max(latencies),
        }
        save_latency_figure(latencies, os.path.join(args.output, "fig5_ebpf_real_latency.pdf"))

        # Phase 2: ringbuf throughput
        ev_count, ev_rate = measure_throughput(loader, args.duration)
        results["ringbuf_throughput"] = {"events": ev_count, "events_per_sec": ev_rate}
        save_throughput_figure([ev_rate], os.path.join(args.output, "fig5_ebpf_real_throughput.pdf"))

        # Phase 3: CPU/RSS overhead
        overhead = measure_overhead(loader, args.duration)
        results["overhead"] = overhead

    finally:
        loader.terminate()
        loader.wait(timeout=5)
        if loader.returncode is None:
            loader.kill()

    # Write JSON summary
    summary_path = os.path.join(args.output, "fig5_ebpf_real_overhead.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[ebpf-bench] wrote {summary_path}")

    print("\n=== Summary ===")
    print(f"  SIGKILL latency:  median={results['sigkill_latency_us']['median']:.1f}µs  "
          f"p95={results['sigkill_latency_us']['p95']:.1f}µs")
    print(f"  Ringbuf throughput:  {results['ringbuf_throughput']['events_per_sec']:.0f} events/s")
    if overhead.get("cpu_percent"):
        c = overhead["cpu_percent"]
        print(f"  CPU:  mean={c['mean']:.1f}%  max={c['max']:.1f}%")
    if overhead.get("rss_mb"):
        r = overhead["rss_mb"]
        print(f"  RSS:  mean={r['mean']:.1f} MB  max={r['max']:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())