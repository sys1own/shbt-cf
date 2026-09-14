//! Floating-point environment control: Denormals-Are-Zero / Flush-To-Zero.
//!
//! Subnormal operands trigger microcode assists on most cores, turning an
//! otherwise deterministic hot loop into one whose timing (and, with some
//! compilers, result ordering) depends on the data. The simulator therefore
//! flushes subnormals to zero on every worker thread at startup. Loss of the
//! subnormal range (`< 2.2e-308`) is irrelevant at the scales the simulator
//! operates in and is documented as part of the numerical contract.
//!
//! * x86_64: MXCSR bits 15 (FTZ) and 6 (DAZ) via `_mm_getcsr` / `_mm_setcsr`.
//! * AArch64: FPCR bit 24 (FZ) and bit 19 (FZ16). AArch64 has a single
//!   flush mode covering inputs and outputs, so DAZ and FTZ are both mapped to FZ.
//!
//! On other targets the calls are no-ops that report `Unsupported`.
//!
//! The flags are thread-local CPU state; call [`enable_flush_to_zero`] once
//! per thread that performs numerics (the Python wrapper does so on its worker
//! pool).

#![allow(unsafe_code)]

// `_mm_getcsr`/`_mm_setcsr` are deprecated in favour of inline asm, but they
// are the canonical, stable MXCSR accessors and lower to the same
// `stmxcsr`/`ldmxcsr` instructions.
#[allow(deprecated)]
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{_mm_getcsr, _mm_setcsr};

/// Snapshot of the subnormal-handling flags on the calling thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlushMode {
    /// Subnormals are processed at full IEEE 754 semantics.
    Ieee,
    /// Subnormal inputs are treated as zero and subnormal results flushed to zero.
    FlushToZero,
    /// The target has no accessible control register for this.
    Unsupported,
}

#[cfg(target_arch = "x86_64")]
const MXCSR_FTZ: u32 = 1 << 15;
#[cfg(target_arch = "x86_64")]
const MXCSR_DAZ: u32 = 1 << 6;

#[cfg(target_arch = "aarch64")]
const FPCR_FZ: u64 = 1 << 24;
#[cfg(target_arch = "aarch64")]
const FPCR_FZ16: u64 = 1 << 19;

/// Reads the raw control register (MXCSR or FPCR) of the calling thread.
#[cfg(target_arch = "x86_64")]
pub fn read_control_register() -> u64 {
    // SAFETY: `_mm_getcsr` only reads MXCSR; SSE is baseline on x86_64.
    #[allow(deprecated)]
    unsafe {
        u64::from(_mm_getcsr())
    }
}

/// Reads the raw control register (MXCSR or FPCR) of the calling thread.
#[cfg(target_arch = "aarch64")]
pub fn read_control_register() -> u64 {
    let fpcr: u64;
    // SAFETY: `mrs` from FPCR is an unprivileged read with no side effects.
    unsafe {
        core::arch::asm!("mrs {}, fpcr", out(reg) fpcr, options(nomem, nostack, preserves_flags));
    }
    fpcr
}

/// Reads the raw control register (MXCSR or FPCR) of the calling thread.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn read_control_register() -> u64 {
    0
}

#[cfg(target_arch = "x86_64")]
fn write_control_register(value: u64) {
    // SAFETY: writing MXCSR only alters the FP control state of this thread.
    // The value originates from `_mm_getcsr` with control bits toggled, so
    // reserved bits are preserved and no #GP fault can occur.
    #[allow(deprecated)]
    unsafe {
        _mm_setcsr(value as u32);
    }
}

#[cfg(target_arch = "aarch64")]
fn write_control_register(value: u64) {
    // SAFETY: `msr fpcr` is an unprivileged write affecting only this thread's
    // FP control state; the value derives from a prior read with only FZ bits
    // changed, so reserved bits are preserved.
    unsafe {
        core::arch::asm!("msr fpcr, {}", in(reg) value, options(nomem, nostack, preserves_flags));
    }
}

/// Current flush mode of the calling thread.
pub fn current_mode() -> FlushMode {
    #[cfg(target_arch = "x86_64")]
    {
        let csr = read_control_register() as u32;
        if csr & (MXCSR_FTZ | MXCSR_DAZ) == (MXCSR_FTZ | MXCSR_DAZ) {
            FlushMode::FlushToZero
        } else {
            FlushMode::Ieee
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if read_control_register() & FPCR_FZ != 0 {
            FlushMode::FlushToZero
        } else {
            FlushMode::Ieee
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        FlushMode::Unsupported
    }
}

/// Sets DAZ + FTZ on the calling thread. Returns the resulting mode.
pub fn enable_flush_to_zero() -> FlushMode {
    #[cfg(target_arch = "x86_64")]
    {
        write_control_register(read_control_register() | u64::from(MXCSR_FTZ | MXCSR_DAZ));
    }
    #[cfg(target_arch = "aarch64")]
    {
        write_control_register(read_control_register() | FPCR_FZ | FPCR_FZ16);
    }
    current_mode()
}

/// Clears DAZ + FTZ, restoring full IEEE 754 subnormal handling.
pub fn disable_flush_to_zero() -> FlushMode {
    #[cfg(target_arch = "x86_64")]
    {
        write_control_register(read_control_register() & !u64::from(MXCSR_FTZ | MXCSR_DAZ));
    }
    #[cfg(target_arch = "aarch64")]
    {
        write_control_register(read_control_register() & !(FPCR_FZ | FPCR_FZ16));
    }
    current_mode()
}

/// RAII guard: enables flush-to-zero for its lifetime and restores the prior
/// register contents on drop.
#[derive(Debug)]
pub struct FlushToZeroGuard {
    saved: u64,
}

impl FlushToZeroGuard {
    /// Enables DAZ/FTZ on the current thread, remembering the previous state.
    pub fn new() -> Self {
        let saved = read_control_register();
        enable_flush_to_zero();
        Self { saved }
    }

    /// Register contents captured before the guard enabled flushing.
    pub fn saved_register(&self) -> u64 {
        self.saved
    }
}

impl Default for FlushToZeroGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for FlushToZeroGuard {
    fn drop(&mut self) {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        write_control_register(self.saved);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[inline(never)]
    fn opaque(x: f64) -> f64 {
        core::hint::black_box(x)
    }

    #[test]
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    fn flush_mode_round_trip_and_effect() {
        let before = read_control_register();
        {
            let guard = FlushToZeroGuard::new();
            assert_eq!(guard.saved_register(), before);
            assert_eq!(current_mode(), FlushMode::FlushToZero);
            // A product landing in the subnormal range must flush to exactly zero.
            let tiny = opaque(f64::MIN_POSITIVE) * opaque(0.5);
            assert_eq!(tiny, 0.0);
            // And a subnormal input must be treated as zero.
            let sub = opaque(f64::from_bits(1)) * opaque(1.0e300);
            assert_eq!(sub, 0.0);
        }
        assert_eq!(read_control_register(), before);
        if current_mode() == FlushMode::Ieee {
            let tiny = opaque(f64::MIN_POSITIVE) * opaque(0.5);
            assert!(tiny > 0.0 && tiny.is_subnormal());
        }
    }

    #[test]
    fn explicit_enable_disable() {
        let before = read_control_register();
        let _ = enable_flush_to_zero();
        let _ = disable_flush_to_zero();
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        assert_eq!(current_mode(), FlushMode::Ieee);
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        write_control_register(before);
        let _ = before;
    }
}
