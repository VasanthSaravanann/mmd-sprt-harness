//! Userspace eBPF verifier: loads the kernel program, attaches kprobes and
//! drains near-miss telemetry from the ringbuffer.
//!
//! Attaches to `__x64_sys_connect` and `__x64_sys_openat` on x86_64.
//! `sys_enter_*` are tracepoint names, not kprobe symbols — the real kernel
//! symbols are `__x64_sys_connect` / `__x64_sys_openat` (or `__arm64_*` on aarch64).
//! See crates/ebpf-verifier-ebpf/src/main.rs for the kernel-side handler.
//!
//! Requires a privileged Linux host with `/sys/kernel/debug/tracing`; it is not
//! built on macOS/dev hosts (see `README.md`).

use std::path::PathBuf;

use aya::maps::ring_buf::RingBuf;
use aya::programs::KProbe;
use aya::{Btf, EbpfLoader};
use aya_log::EbpfLogger;
use clap::Parser;
use log::{info, warn};

/// Must match `NearMissEvent` in `ebpf-verifier-ebpf/src/main.rs`.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct NearMissEvent {
    pid: u32,
    tid: u32,
    kind: u32,
    tier: u32,
    token_digest: u64,
    ts_ns: u64,
}

const EVENT_SIZE: usize = std::mem::size_of::<NearMissEvent>();

#[derive(Parser)]
struct Opt {
    /// Path to the compiled eBPF object.
    #[arg(short, long)]
    path: PathBuf,
    /// Exit after this many telemetry events (0 = run forever).
    #[arg(short, long, default_value_t = 0)]
    count: usize,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let opt = Opt::parse();

    let bytes = std::fs::read(&opt.path)
        .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", opt.path.display()))?;

    // BTF is optional; attach proceeds without it if unavailable.
    // aya 0.13: Ebpf::load no longer accepts BTF; use EbpfLoader instead.
    let btf = Btf::from_sys_fs().ok();
    let mut ebpf = EbpfLoader::new().btf(btf.as_ref()).load(&bytes)?;
    if let Err(e) = EbpfLogger::init(&mut ebpf) {
        warn!("eBPF logger init failed (continuing): {e}");
    }

    // `sys_enter_*` tracepoints don't exist as kprobe symbols; the real kernel
    // function names for syscall handlers are `__x64_sys_connect`,
    // `__x64_sys_openat` (x86_64) or `__arm64_sys_*` (aarch64).  The program
    // names (`#[kprobe]` in the ebpf crate) are arbitrary eBPF function labels.
    for (prog_name, hook) in [
        ("sys_enter_connect", "__x64_sys_connect"),
        ("sys_enter_openat", "__x64_sys_openat"),
    ] {
        let prog: &mut KProbe = ebpf
            .program_mut(prog_name)
            .ok_or_else(|| anyhow::anyhow!("program {prog_name} not found"))?
            .try_into()?;
        prog.load()?;
        // aya 0.13: KProbe::attach requires an offset (0 = function entry).
        prog.attach(hook, 0)?;
        info!("attached kprobe {hook}");
    }

    info!("verifier attached; draining ringbuf");
    let mut rb = RingBuf::try_from(
        ebpf.take_map("NEAR_MISS_RINGBUF")
            .ok_or_else(|| anyhow::anyhow!("NEAR_MISS_RINGBUF map not found"))?,
    )?;

    let mut seen = 0usize;
    loop {
        match rb.next() {
            Some(item) => {
                if item.len() >= EVENT_SIZE {
                    let raw = item.as_ref();
                    let ev: NearMissEvent = unsafe {
                        std::ptr::read_unaligned(raw.as_ptr() as *const NearMissEvent)
                    };
                    info!(
                        "event pid={} tid={} kind={} tier={} digest=0x{:x}",
                        ev.pid, ev.tid, ev.kind, ev.tier, ev.token_digest
                    );
                    seen += 1;
                    if opt.count != 0 && seen >= opt.count {
                        info!("reached event count {seen}; exiting");
                        break;
                    }
                }
            }
            None => {
                // No event pending: block briefly instead of burning a core in a
                // hot spin (the previous implementation called yield_now() in a
                // tight loop and pinned a CPU at 100%).
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
    Ok(())
}
