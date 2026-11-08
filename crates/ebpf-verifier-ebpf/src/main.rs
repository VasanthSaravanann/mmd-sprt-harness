//! In-kernel eBPF verifier program (aya-ebpf).
//!
//! * `sys_enter_connect` kprobe  — telemetry (Tier 2 near-miss).
//! * `sys_enter_openat`  kprobe  — **honeypot interceptor**.  `openat` carries
//!   the user-space `filename` pointer, so this is the correct hook to detect
//!   direct path access to `/tmp/canary_token.key`.  On a match the program
//!   fires `bpf_send_signal(SIGKILL)` directly in-kernel (<10µs) and blocks the
//!   call by returning a non-zero error.
//! * Near-miss events are emitted to userspace over a `BPF_MAP_TYPE_RINGBUF`
//!   (zero-copy telemetry, replacing disk-bound `near_misses.jsonl`).
//!
//! **Build note:** this crate targets `bpfel-unknown-none` on a Linux host with
//! a nightly toolchain (see `README.md`).  It is excluded from the workspace's
//! default build and is NOT compiled on macOS/dev hosts.

#![no_std]
#![no_main]

use aya_ebpf::bindings::pt_regs;
use aya_ebpf::helpers::{bpf_ktime_get_ns, bpf_probe_read_user_str_bytes, bpf_send_signal};
use aya_ebpf::macros::{kprobe, map};
use aya_ebpf::maps::RingBuf;
use aya_ebpf::programs::KProbeContext;
use aya_log_ebpf::info;

/// Telemetry event emitted to userspace (must match the userspace mirror).
#[repr(C)]
#[derive(Copy, Clone)]
pub struct NearMissEvent {
    pub pid: u32,
    pub tid: u32,
    pub kind: u32,         // 0 = connect near-miss, 1 = honeypot trip
    pub tier: u32,         // 1 = immediate kill, 2 = deferred
    pub token_digest: u64, // FNV-1a digest of the accessed path/domain
    pub ts_ns: u64,
}

/// Zero-copy telemetry stream to userspace.
#[map(name = "NEAR_MISS_RINGBUF")]
static RING: RingBuf = RingBuf::with_byte_size(1024 * 1024, 0);

/// Honeypot path compared byte-wise (keep in sync with userspace).
const CANARY_PATH: &[u8] = b"/tmp/canary_token.key";
const SIGKILL: u32 = 9;

#[inline(always)]
fn emit(ctx: &KProbeContext, kind: u32, tier: u32, digest: u64) {
    let event = NearMissEvent {
        pid: ctx.pid(),
        tid: ctx.tid(),
        kind,
        tier,
        token_digest: digest,
        ts_ns: unsafe { bpf_ktime_get_ns() },
    };
    // Reserve + write + submit in one call; drops the event if the ring is full.
    let _ = RING.output(&event, 0);
}

#[inline(always)]
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        i += 1;
    }
    hash
}

/// Read up to `CANARY_PATH.len()` bytes of the `openat` filename argument.
#[inline(always)]
unsafe fn read_openat_filename(ctx: &KProbeContext) -> [u8; CANARY_PATH.len()] {
    let regs: &pt_regs = ctx.regs();
    let arg_ptr = regs.si as *const u8; // x86_64 syscall arg 1 (filename)
    let mut buf = [0u8; CANARY_PATH.len()];
    let _ = bpf_probe_read_user_str_bytes(arg_ptr, &mut buf);
    buf
}

/// Kprobe on `sys_enter_connect` — Tier 2 telemetry.
#[kprobe]
pub fn sys_enter_connect(ctx: KProbeContext) -> u32 {
    info!(&ctx, "connect pid={} tid={}", ctx.pid(), ctx.tid());
    emit(&ctx, 0, 2, 0);
    0
}

/// Kprobe on `sys_enter_openat` — Tier 1 honeypot interceptor.
#[kprobe]
pub fn sys_enter_openat(ctx: KProbeContext) -> u32 {
    let filename = unsafe { read_openat_filename(&ctx) };
    if filename == *CANARY_PATH {
        let digest = fnv1a(&filename);
        emit(&ctx, 1, 1, digest);
        // Kill the offending task from the kernel (SIGKILL), then abort the call.
        unsafe { bpf_send_signal(SIGKILL) };
        return 1; // non-zero -> make the syscall fail fast
    }
    emit(&ctx, 0, 2, fnv1a(&filename));
    0
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    // eBPF programs cannot unwind; park the (already-verifier-rejected) path.
    loop {}
}
