//! In-kernel eBPF verifier program (aya-ebpf).
//!
//! Kprobes attach to kernel syscall-entry functions; the symbol names on x86_64
//! are `__x64_sys_connect` and `__x64_sys_openat` (the userspace loader in
//! `ebpf-verifier` uses those names).  The `#[kprobe]` macro labels below are
//! eBPF function identifiers — they do NOT need to match the kernel symbol.
//!
//! * `__x64_sys_connect` kprobe — telemetry (Tier 2 near-miss).
//! * `__x64_sys_openat`  kprobe — **honeypot interceptor**.  `openat` carries
//!   the user-space `filename` pointer as its second register argument (`regs.si`
//!   on x86_64), so this is the correct hook to detect direct path access to
//!   `/tmp/canary_token.key`.  On a match the program fires
//!   `bpf_send_signal(SIGKILL)` directly in-kernel (<10µs).
//!   Note: a kprobe cannot veto the syscall; the mitigation is the signal, not
//!   the return code.
//! * Near-miss events are emitted to userspace over a `BPF_MAP_TYPE_RINGBUF`
//!   (zero-copy telemetry, replacing disk-bound `near_misses.jsonl`).
//!
//! **Build note:** this crate targets `bpfel-unknown-none` on a Linux host with
//! a nightly toolchain (see `README.md`).  It is excluded from the workspace's
//! default build and is NOT compiled on macOS/dev hosts.

#![no_std]
#![no_main]

// aya-ebpf 0.1.1 renamed KProbeContext to ProbeContext, and the regs() method
// is now a field (ctx.regs).  EbpfContext provides pid() / tgid().
use aya_ebpf::bindings::pt_regs;
use aya_ebpf::helpers::{bpf_ktime_get_ns, bpf_probe_read_user_str_bytes, bpf_send_signal};
use aya_ebpf::macros::{kprobe, map};
use aya_ebpf::maps::RingBuf;
use aya_ebpf::programs::ProbeContext;
use aya_ebpf::EbpfContext;
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
/// `openat`/`open` filename buffers are compared against the path **including**
/// its NUL terminator so that `probe_read_user_str_bytes` output (always
/// NUL-terminated) matches exactly and `prefix/openat2` variants cannot slip
/// past a plain byte-equality check.
const CANARY_PATH_Z: &[u8] = b"/tmp/canary_token.key\0";
/// Buffer bound for the syscall filename read.  Must be `<= 512` and a power of
/// two so the eBPF verifier accepts a constant-size, bounded comparison loop.
const FILENAME_BUF: usize = 64;
const SIGKILL: u32 = 9;

#[inline(always)]
// BPF convention: pid() = thread id, tgid() = process id.
fn emit(ctx: &ProbeContext, kind: u32, tier: u32, digest: u64) {
    let event = NearMissEvent {
        pid: ctx.tgid(),  // process id
        tid: ctx.pid(),   // thread id
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
    // Unrolled with a constant bound so the verifier can prove termination.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        i += 1;
    }
    hash
}

/// Read the `openat` filename argument (x86_64 syscall arg 1 = `rsi`).
/// The x86_64 `pt_regs` struct uses the field name `si` (not `rsi`).
#[inline(always)]
unsafe fn read_openat_filename(ctx: &ProbeContext) -> [u8; FILENAME_BUF] {
    let regs: &pt_regs = &*ctx.regs;
    let arg_ptr = regs.si as *const u8;
    let mut buf = [0u8; FILENAME_BUF];
    let _ = bpf_probe_read_user_str_bytes(arg_ptr, &mut buf);
    buf
}

/// Return true if the (NUL-terminated) read buffer equals `CANARY_PATH`.
#[inline(always)]
fn is_canary_path(buf: &[u8; FILENAME_BUF]) -> bool {
    // Compare against the NUL-terminated canary, bounded by FILENAME_BUF.
    let mut i = 0;
    while i < FILENAME_BUF {
        let expected = if i < CANARY_PATH_Z.len() { CANARY_PATH_Z[i] } else { 0 };
        if buf[i] != expected {
            return false;
        }
        if buf[i] == 0 {
            break;
        }
        i += 1;
    }
    // Reject prefix attacks: a canary prefix followed by extra bytes must not
    // match.  `CANARY_PATH_Z` includes its NUL, so the comparison above already
    // enforces that buf[CANARY_PATH.len()] == 0.
    true
}

/// Kprobe on `__x64_sys_connect` — Tier 2 telemetry.
///
/// The userspace loader attaches this program to the kernel symbol
/// `__x64_sys_connect` (the function name here is just the eBPF program label).
#[kprobe]
pub fn sys_enter_connect(ctx: ProbeContext) -> u32 {
    info!(&ctx, "connect pid={} tid={}", ctx.tgid(), ctx.pid());
    emit(&ctx, 0, 2, 0);
    0
}

/// Kprobe on `__x64_sys_openat` — Tier 1 honeypot interceptor.
///
/// On x86_64, at the entry of `__x64_sys_openat`, `regs.si` contains the
/// user-space filename pointer (syscall arg 1).  We read the path into a
/// bounded buffer and compare it against the canary path.
#[kprobe]
pub fn sys_enter_openat(ctx: ProbeContext) -> u32 {
    let filename = unsafe { read_openat_filename(&ctx) };
    if is_canary_path(&filename) {
        let digest = fnv1a(CANARY_PATH);
        emit(&ctx, 1, 1, digest);
        // Kill the offending task from the kernel (SIGKILL).
        //
        // IMPORTANT: kprobes are debug hooks and **cannot abort the syscall**;
        // a non-zero return value here is ignored by the kernel.  The mitigation
        // is the SIGKILL itself, not a return-code veto.  (Blocking the syscall
        // would require an LSM BPF hook / `bpf_lsm` program.)
        unsafe { bpf_send_signal(SIGKILL) };
        return 1;
    }
    emit(&ctx, 0, 2, 0);
    0
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    // eBPF programs cannot unwind; park the (already-verifier-rejected) path.
    loop {}
}
