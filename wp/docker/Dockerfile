# Docker image for the eBPF verifier measurement environment.
# Build and run:
#   docker build -t ebpf-bench wp/docker
#   docker run --privileged -it ebpf-bench
#
# For gVisor (runsc):
#   docker run --runtime=runsc --privileged -it ebpf-bench

FROM rust:bookworm AS builder

RUN apt-get update && apt-get install -y \
    clang llvm libbpf-dev linux-headers-$(uname -r) \
    stress-ng python3-pip

RUN rustup toolchain install nightly --component rust-src
RUN cargo install bpf-linker

WORKDIR /app
COPY . .

RUN cargo +nightly build -p ebpf-verifier-ebpf --release \
    --target bpfel-unknown-none -Z build-std=core
RUN cargo build -p ebpf-verifier --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y stress-ng python3 python3-pip ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip3 install matplotlib numpy scipy scikit-learn psutil
COPY --from=builder /app/target /app/target
COPY --from=builder /app/benches /app/benches
WORKDIR /app

ENTRYPOINT ["/usr/bin/python3", "benches/eval_ebpf_bench.py", \
    "--ebpf-obj", "target/bpfel-unknown-none/release/ebpf-verifier-ebpf", \
    "--output", "benches/ebpf-bench", "--stress"]