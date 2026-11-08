# Docker + gVisor deployment for the eBPF verifier measurement environment.
#
# NOTE: eBPF cannot load inside gVisor (gVisor intercepts syscalls; bpf() is
# unavailable).  This image is for NATIVE kernel measurement.  For gVisor
# comparison, run the image under --runtime=runsc but only the Python harness
# benchmarks will work (Gate 0 / SPRT / IPW).
#
# Build from the WORKSPACE ROOT, not from wp/docker/:
#   docker build -t ebpf-bench -f wp/docker/Dockerfile .
#
# Run (native eBPF measurement — requires --privileged):
#   docker run --privileged -it --rm ebpf-bench
#
# Run (gVisor — only Python harness benches work):
#   docker run --runtime=runsc --rm ebpf-bench \
#     .venv/bin/python benches/run_all_benches.sh

FROM rust:bookworm AS builder

RUN apt-get update && apt-get install -y \
    clang llvm libbpf-dev stress-ng python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN rustup toolchain install nightly --component rust-src
RUN cargo install bpf-linker

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/
COPY benches/ ./benches/
COPY tests/ ./tests/
COPY harness/ ./harness/
COPY paper/ ./paper/
COPY wp/ ./wp/

RUN cargo +nightly build -p ebpf-verifier-ebpf --release \
    --target bpfel-unknown-none -Z build-std=core
RUN cargo build -p ebpf-verifier --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y \
    stress-ng python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --break-system-packages matplotlib numpy scipy scikit-learn psutil

COPY --from=builder /app/target /app/target
COPY --from=builder /app/benches /app/benches
COPY --from=builder /app/harness /app/harness
COPY --from=builder /app/tests /app/tests
COPY --from=builder /app/paper /app/paper
COPY --from=builder /app/wp /app/wp
WORKDIR /app

# RUST_LOG must be set for aya-log to emit telemetry events to stderr.
ENV RUST_LOG=info

ENTRYPOINT ["python3", "benches/eval_ebpf_bench.py", \
    "--ebpf-obj", "target/bpfel-unknown-none/release/ebpf-verifier-ebpf", \
    "--output", "benches/ebpf-bench", "--stress"]