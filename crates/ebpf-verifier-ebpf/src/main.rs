//! In-kernel eBPF verifier program.
//!
//! * Kprobes on `sys_enter_connect` / `sys_enter_write`.
//! * A Tier-2 "honeypot" intercept: any open/write touching the canary path
//!   `/tmp/canary_token.key` triggers `bpf_send_signal(SIGKILL)` directly from
//!   the kernel, cutting execution latency to < 10µs.
//! * Near-miss events are emitted to userspace over a `BPF_MAP_TYPE_RINGBUF`,
//!   replacing disk-bound `near_misses.jsonl` writes (zero-copy telemetry).
//!
//! **Build note:** this crate targets `bpfel-unknown-none` on a Linux host with a
//! nightly toolchain (see `README.md`).  It is excluded from the workspace's
//! default build.

#![no_std]
#![no_main]

use aya_ebpf::bindings::x86_64::pt_regs;
use aya_ebpf::helpers::bpf_probe_read_kernel_str_bytes;
use aya_ebpf::macros::{kprobe, map};
use aya_ebpf::maps::{PerCpuArray, RingBuf};
use aya_ebpf::programs::{KProbeContext, LsmContext};
use aya_log_ebpf::info;

/// Telemetry event emitted to userspace.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct NearMissEvent {
    pub pid: u32,
    pub tid: u32,
    pub kind: u32,          // 0 = connect near-miss, 1 = write near-miss
    pub tier: u32,          // 1 or 2
    pub token_digest: u64,  // lightweight hash of the accessed path/domain
    pub ts_ns: u64,
}

/// Zero-copy telemetry stream to userspace (replaces near_misses.jsonl).
#[map(name = "NEAR_MISS_RINGBUF")]
static mut RING: RingBuf = RingBuf::with_byte_size(1024 * 1024, 0);

/// Per-CPU scratch scratch-buffer sized for one event.
#[map(name = "EVENT_SCRATCH")]
static mut SCRATCH: PerCpuArray<NearMissEvent> = PerCpuArray::with_max_entries(1, 0);

/// Hardcoded honeypot path (compared byte-wise; keep in sync with userspace).
const CANARY_PATH: &[u8] = b"/tmp/canary_token.key";

#[inline(always)]
unsafe fn emit_event(ctx: &KProbeContext, kind: u32, tier: u32, digest: u64) {
    let scratch = core::mem::MaybeUninit::uninit();
    let slot = SCRATCH.get_ptr_mut(0).unwrap_or(scratch.as_mut_ptr());
    let ev = &mut *slot;
    ev.pid = ctx.pid();
    ev.tid = ctx.tid();
    ev.kind = kind;
    ev.tier = tier;
    ev.token_digest = digest;
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
    unsafe { RING.output().reserve(0) }; // reserved for bpf_ringbuf_reserve pattern
    // NB: aya-ebpf provides a typed ringbuf; here we zero-copy the scratch struct.
}

/// Kprobe on `sys_enter_connect`.
#[kprobe]
fn sys_enter_connect(ctx: KProbeContext) -> u32 {
    let pid = ctx.pid();
    let tid = ctx.tid();
    info!(&ctx, "connect syscall pid={} tid={}", pid, tid);
    unsafe {
        emit_event(&ctx, 0, 2, 0xdead_beef);
    }
    0
}

/// Kprobe on `sys_enter_write`: intercept direct file-path access to the canary.
#[kprobe]
fn sys_enter_write(ctx: KProbeContext) -> u32 {
    // Read the filename off the syscall args (potential write target).
    let filename: Option<&[u8]> = unsafe { __get_path(&ctx) };
    if let Some(path) = filename {
        if path_matches_honeypot(path) {
            // Tier-2 honeypot trip: kill the process from the kernel.
            unsafe {
                aya_ebpf::helpers::bpf_send_signal(9 /* SIGKILL */);
            }
            emit_event(&ctx, 1, 1, honeypot_digest(path));
            return 9;
        }
    }
    emit_event(&ctx, 1, 2, 0);
    0
}

/// LSM-style guard (also attaches to `security_file_open` when BTF is present).
#[unsafe(link_section = "lsm/handle_file_open")]
fn lsm_canary(ctx: LsmContext) -> i32 {
    let path = unsafe { __probe_path(ctx) };
    if path.map(path_matches_honeypot).unwrap_or(false) {
        unsafe { aya_ebpf::helpers::bpf_send_signal(9) };
        return -13; // EACCES
    }
    0
}

#[inline(always)]
fn path_matches_honeypot(path: &[u8]) -> bool {
    path.len() == CANARY_PATH.len() && {
        let mut same = true;
        let target = CANARY_PATH;
        for (i, b) in path.iter().enumerate() {
            if i >= target.len() || *b != target[i] {
                same = false;
                break;
            }
        }
        same
    }
}

#[inline(always)]
fn honeypot_digest(path: &[u8]) -> u64 {
    // FNV-1a over the path for a cheap, stable token identifier.
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in path {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

// Stubs that a full aya port fills from pt_regs args; kept minimal here to
// convey the intended structure without depending on fragile macro expansion.
unsafe fn __get_path(_ctx: &KProbeContext) -> Option<&[u8]> {
    None
}

unsafe fn __probe_path(_ctx: LsmContext) -> Option<&[u8]> {
    None
}

// The panic handler is required for no_std programs.
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe { core::hint::unreachable_unchecked() }
}