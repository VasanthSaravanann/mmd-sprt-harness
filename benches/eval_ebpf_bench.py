#!/usr/bin/env python3
"""Real eBPF verifier benchmark (runs on Linux, requires root).

Measures:
  1. Canary-open-to-SIGKILL latency (µs) — forking a child that touches the
     canary file; the parent times how long the child survives.
  2. Ringbuffer telemetry throughput (events/sec) — counts ``event pid=`` lines
     from the loader's stderr while firing connect() syscalls.
  3. CPU and RSS overhead of the userspace loader under sustained load.

Usage:
  # Build the kernel program first (on the Linux build host):
  cargo +nightly build -p ebpf-verifier-ebpf --release \\
      --target bpfel-unknown-none -Z build-std=core

  # Run the benchmark (root required for eBPF):
  sudo .venv/bin/python benches/eval_ebpf_bench.py \\
      --ebpf-obj target/bpfel-unknown-none/release/ebpf-verifier-ebpf \\
      --output paper/figures

Output:
  - paper/figures/fig5_ebpf_real_latency.pdf
  - paper/figures/fig5_ebpf_real_throughput.pdf
  - paper/figures/fig5_ebpf_real_overhead.json
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from threading import Thread
from typing import Iterator, List, Optional, Tuple

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


def parse_args() -> Namespace:
    p = ArgumentParser(description="Real eBPF verifier benchmark")
    p.add_argument("--ebpf-obj", required=True,
                   help="Path to compiled ebpf-verifier-ebpf .o")
    p.add_argument("--output", default="paper/figures",
                   help="Output directory for figures & JSON")
    p.add_argument("--count", type=int, default=150,
                   help="Number of canary-touch iterations for latency measurement")
    p.add_argument("--duration", type=float, default=10.0,
                   help="Ringbuf throughput measurement duration (seconds)")
    p.add_argument("--stress", action="store_true",
                   help="Run stress-ng CPU load during latency measurement")
    return p.parse_args()


def check_prereqs() -> None:
    if os.geteuid() != 0:
        print("WARNING: not running as root — eBPF program loading will fail.",
              file=sys.stderr)
    Path(CANARY_PATH).write_bytes(b"canary")  # create the honeypot target
    if not HAS_PSUTIL:
        print("INFO: psutil not installed; CPU/RSS measurement disabled.",
              file=sys.stderr)
        print("  install: pip install psutil", file=sys.stderr)


# ---------------------------------------------------------------------------
#  Helper: drain loader stderr in a thread (prevents pipe deadlock)
# ---------------------------------------------------------------------------

def _drain_stderr(stream: Iterator[bytes]) -> int:
    """Count lines containing 'event pid='; return total."""
    count = 0
    for line in stream:
        if b"event pid=" in line:
            count += 1
    return count


# ---------------------------------------------------------------------------
#  1. Canary-open-to-SIGKILL latency  (via forked child)
# ---------------------------------------------------------------------------

def _child_touch() -> None:
    """Child process: open the canary file.  The kernel kills us with SIGKILL."""
    # SIGKILL cannot be caught/blocked, so this call will never return.
    try:
        with open(CANARY_PATH, "rb"):
            pass
    except (FileNotFoundError, PermissionError, OSError):
        pass
    # If we somehow survive, make sure we exit loudly.
    os._exit(99)


def measure_latency(loader_pid: int, n: int,
                    args: Namespace) -> List[float]:
    """Measure SIGKILL latency by forking children and timing their lifespan.

    Returns wall-clock time (µs) from child creation to SIGKILL delivery.
    """
    print(f"[ebpf-bench] Measuring SIGKILL latency ({n} samples)...")

    stress: Optional[subprocess.Popen] = None
    if args.stress:
        try:
            n_cpu = os.cpu_count() or 4
            stress = subprocess.Popen(
                ["stress-ng", "--cpu", str(n_cpu),
                 "--timeout", f"{int(math.ceil(n * 0.05 + 5))}s"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            print(f"  stress-ng running (pid={stress.pid})")
        except FileNotFoundError:
            print("  (stress-ng not found; skipping CPU load)")

    latencies: List[float] = []
    for i in range(n):
        start = time.perf_counter()
        pid = os.fork()
        if pid == 0:
            # Child: touch the canary and be killed
            _child_touch()
        else:
            # Parent: wait for the child to be killed by SIGKILL (signal 9)
            wpid, status = os.waitpid(pid, 0)
            elapsed = (time.perf_counter() - start) * 1e6
            if os.WIFSIGNALED(status) and os.WTERMSIG(status) == 9:
                latencies.append(elapsed)
            else:
                print(f"  WARNING: child {pid} exited with status {status} "
                      f"(expected SIGKILL=9)", file=sys.stderr)
        time.sleep(0.01)  # small inter-touch gap

    if stress:
        stress.terminate()
        try:
            stress.wait(timeout=5)
        except subprocess.TimeoutExpired:
            stress.kill()

    print(f"  samples={len(latencies)}  median={_p(latencies, 50):.1f}µs  "
          f"p95={_p(latencies, 95):.1f}µs  max={max(latencies):.1f}µs")
    return latencies


# ---------------------------------------------------------------------------
#  2. Ringbuffer telemetry throughput
# ---------------------------------------------------------------------------

def _connect_storm(duration: float) -> None:
    """Fire connect() calls in a tight loop for `duration` seconds."""
    import socket
    end = time.perf_counter() + duration
    while time.perf_counter() < end:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.001)
            s.connect(("127.0.0.1", 1))
        except (ConnectionRefusedError, OSError, socket.timeout):
            pass
        finally:
            s.close()


def measure_throughput(duration: float) -> Tuple[int, float]:
    """Count ringbuf events from the loader's stderr while firing connects."""
    print(f"[ebpf-bench] Measuring ringbuf throughput ({duration}s)...")

    # Fire connect syscalls in a background thread while we count events.
    storm = Thread(target=_connect_storm, args=(duration,), daemon=True)
    storm.start()

    # The event counting is done by the stderr drain thread which was started
    # before the storm.  We just wait for the storm to finish, then give the
    # drain thread a moment to catch up.
    storm.join()
    time.sleep(0.5)

    # The count is accumulated in _stderr_count (set by main()).
    return None  # placeholder — count is returned via _stderr_count


# ---------------------------------------------------------------------------
#  3. CPU / RSS overhead
# ---------------------------------------------------------------------------

def measure_overhead(proc: psutil.Process,
                     duration: float) -> dict:
    """Sample CPU% and RSS of the loader process."""
    if not HAS_PSUTIL:
        return {"cpu_percent": None, "rss_mb": None}
    print(f"[ebpf-bench] Measuring CPU/RSS ({duration}s)...")
    cpu_samples: List[float] = []
    rss_samples: List[float] = []
    end = time.perf_counter() + duration
    while time.perf_counter() < end:
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
    ax.set_title(f"eBPF SIGKILL latency "
                 f"(n={len(latencies)}, median={_p(latencies, 50):.1f}µs)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"[ebpf-bench] wrote {path}")


def save_throughput_figure(events_per_sec: float, path: str) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["ringbuf events/s"], [events_per_sec], color="#81c784")
    ax.set_ylabel("events / sec")
    ax.set_title(f"eBPF ringbuf throughput ({events_per_sec:.0f} events/s)")
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

    # Locate the userspace loader binary.
    bin_path = Path(__file__).resolve().parent.parent / "target" / "release" / "ebpf-verifier"
    if not bin_path.exists():
        print(f"ERROR: loader binary not found: {bin_path}. "
              f"Build it first: cargo build -p ebpf-verifier --release",
              file=sys.stderr)
        return 1

    print(f"[ebpf-bench] Starting loader: {bin_path} --path {ebpf_obj}")

    # Use a temp file for stderr so we can count events without pipe deadlock.
    # stdout is mostly quiet; we redirect it to /dev/null.
    stderr_tmp = tempfile.NamedTemporaryFile(prefix="ebpf-loader-", suffix=".log",
                                             delete=False, mode="w+")
    stderr_path = stderr_tmp.name

    loader = subprocess.Popen(
        [str(bin_path), "--path", str(ebpf_obj)],
        stdout=subprocess.DEVNULL,
        stderr=stderr_tmp,
    )
    time.sleep(1.5)  # wait for kprobe attachment + startup

    if loader.poll() is not None:
        print(f"ERROR: loader exited prematurely (rc={loader.returncode})",
              file=sys.stderr)
        stderr_tmp.close()
        with open(stderr_path) as f:
            sys.stderr.write(f.read())
        return 1

    results: dict = {}
    stderr_tmp.close()

    try:
        # Phase 1: enforcement latency (fork child approach)
        latencies = measure_latency(loader.pid, args.count, args)
        results["sigkill_latency_us"] = {
            "n": len(latencies),
            "median": _p(latencies, 50),
            "p95": _p(latencies, 95),
            "p99": _p(latencies, 99),
            "min": min(latencies) if latencies else 0,
            "max": max(latencies) if latencies else 0,
        }
        save_latency_figure(latencies,
                            os.path.join(args.output, "fig5_ebpf_real_latency.pdf"))

        # Phase 2: ringbuf throughput — fire connects, then count event lines.
        print(f"[ebpf-bench] Measuring ringbuf throughput ({args.duration}s)...")
        # Truncate the stderr log so we count only events during the storm.
        open(stderr_path, "w").close()
        storm = Thread(target=_connect_storm, args=(args.duration,), daemon=True)
        storm.start()
        storm.join()
        time.sleep(0.5)  # let ringbuf drain

        # Count event lines from the loader's stderr.
        event_count = 0
        with open(stderr_path) as f:
            for line in f:
                if "event pid=" in line:
                    event_count += 1
        ev_rate = event_count / args.duration
        results["ringbuf_throughput"] = {
            "events": event_count,
            "events_per_sec": ev_rate,
        }
        save_throughput_figure(ev_rate,
                               os.path.join(args.output, "fig5_ebpf_real_throughput.pdf"))

        # Phase 3: CPU/RSS overhead
        if HAS_PSUTIL:
            overhead = measure_overhead(psutil.Process(loader.pid), args.duration)
            results["overhead"] = overhead
        else:
            results["overhead"] = {"cpu_percent": None, "rss_mb": None}

    finally:
        loader.terminate()
        try:
            loader.wait(timeout=5)
        except subprocess.TimeoutExpired:
            loader.kill()
            loader.wait()
        os.unlink(stderr_path)

    # Write JSON summary
    summary_path = os.path.join(args.output, "fig5_ebpf_real_overhead.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[ebpf-bench] wrote {summary_path}")

    print("\n=== Summary ===")
    lat = results["sigkill_latency_us"]
    print(f"  SIGKILL latency:  median={lat['median']:.1f}µs  "
          f"p95={lat['p95']:.1f}µs  p99={lat['p99']:.1f}µs")
    print(f"  Ringbuf throughput:  {results['ringbuf_throughput']['events_per_sec']:.0f} events/s")
    if results["overhead"].get("cpu_percent") and results["overhead"]["cpu_percent"]["mean"] is not None:
        c = results["overhead"]["cpu_percent"]
        print(f"  CPU:  mean={c['mean']:.1f}%  max={c['max']:.1f}%")
    if results["overhead"].get("rss_mb") and results["overhead"]["rss_mb"]["mean"] is not None:
        r = results["overhead"]["rss_mb"]
        print(f"  RSS:  mean={r['mean']:.1f} MB  max={r['max']:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())