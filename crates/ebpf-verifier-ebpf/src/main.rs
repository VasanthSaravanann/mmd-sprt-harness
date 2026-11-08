//! In-kernel eBPF verifier program (aya-ebpf).
//!
//! Kprobes attach to kernel syscall-entry functions; the symbol names on x86_64
//! are `__x64_sys_connect` and `__x64_sys_openat` (the userspace loader in
//! `ebpf-verifier` uses those names).  The `#[kprobe]` macro labels below are
//! eBPF function identifiers — they do NOT need to match the kernel symbol.
//!
//! * `__x64_sys_connect` kprobe — telemetry (Tier 2 near-miss).
//! * Three honeypot kprobes for `/tmp/canary_token.key`:
//!   - `__x64_sys_openat`  — `openat` syscall (glibc, most Linux programs)
//!   - `__x64_sys_open`    — `open` syscall (musl, Alpine, busybox)
//!   - `__x64_sys_openat2` — `openat2` syscall (newer kernels, 5.6+)
//!   On a match the program fires `bpf_send_signal(SIGKILL)` directly
//!   in-kernel (<10µs).  Note: a kprobe cannot veto the syscall; the
//!   mitigation is the signal, not the return code.
//! * All three open variants use the same inner-pt_regs indirection:
//!   outer.rdi → inner pt_regs, then inner.rdi (=filename) for `open` or
//!   inner.rsi (=filename) for `openat`/`openat2`.
//!   (aya-ebpf bindings use `rdi`/`rsi`, not the kernel header names `di`/`si`.)
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
use aya_ebpf::helpers::{
    bpf_ktime_get_ns, bpf_probe_read_kernel, bpf_probe_read_user_str_bytes, bpf_send_signal,
};
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

/// Read the inner pt_regs from the outer kprobe context.
/// All x86_64 syscall handlers (`__x64_sys_*`) receive a `struct pt_regs *`
/// in `rdi` pointing to the user's saved registers.  We dereference that
/// to get the actual syscall arguments.
#[inline(always)]
unsafe fn inner_pt_regs(ctx: &ProbeContext) -> pt_regs {
    let outer: &pt_regs = &*ctx.regs;
    let inner_ptr = outer.rdi as *const pt_regs;
    bpf_probe_read_kernel(inner_ptr).unwrap_or(core::mem::zeroed())
}

/// Read filename from `inner.rsi` — used by `openat` (glibc) and `openat2`.
#[inline(always)]
unsafe fn read_filename_from_rsi(ctx: &ProbeContext) -> [u8; FILENAME_BUF] {
    let inner: pt_regs = inner_pt_regs(ctx);
    let filename_ptr = inner.rsi as *const u8;
    let mut buf = [0u8; FILENAME_BUF];
    let _ = bpf_probe_read_user_str_bytes(filename_ptr, &mut buf);
    buf
}

/// Read filename from `inner.rdi` — used by `open` (musl, Alpine).
/// (aya-ebpf bindings use `rdi`, not the kernel header name `di`.)
#[inline(always)]
unsafe fn read_filename_from_di(ctx: &ProbeContext) -> [u8; FILENAME_BUF] {
    let inner: pt_regs = inner_pt_regs(ctx);
    let filename_ptr = inner.rdi as *const u8;
    let mut buf = [0u8; FILENAME_BUF];
    let _ = bpf_probe_read_user_str_bytes(filename_ptr, &mut buf);
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

/// Kprobe on `__x64_sys_openat` — Tier 1 honeypot interceptor (glibc path).
///
/// `openat` carries the filename in `rsi` of the inner pt_regs.
#[kprobe]
pub fn sys_enter_openat(ctx: ProbeContext) -> u32 {
    let filename = unsafe { read_filename_from_rsi(&ctx) };
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

/// Kprobe on `__x64_sys_open` — Tier 1 honeypot interceptor (musl / Alpine path).
///
/// `open` carries the filename in `di` (not `rsi` like `openat`).
#[kprobe]
pub fn sys_enter_open(ctx: ProbeContext) -> u32 {
    let filename = unsafe { read_filename_from_di(&ctx) };
    if is_canary_path(&filename) {
        let digest = fnv1a(CANARY_PATH);
        emit(&ctx, 1, 1, digest);
        unsafe { bpf_send_signal(SIGKILL) };
        return 1;
    }
    emit(&ctx, 0, 2, 0);
    0
}

/// Kprobe on `__x64_sys_openat2` — Tier 1 honeypot interceptor (newer kernels).
///
/// `openat2` uses the same syscall ABI as `openat` for the filename (in `rsi`).
#[kprobe]
pub fn sys_enter_openat2(ctx: ProbeContext) -> u32 {
    let filename = unsafe { read_filename_from_rsi(&ctx) };
    if is_canary_path(&filename) {
        let digest = fnv1a(CANARY_PATH);
        emit(&ctx, 1, 1, digest);
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
