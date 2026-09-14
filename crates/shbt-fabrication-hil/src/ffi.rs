//! C-ABI exports for non-Python hosts (CFI/CDI bridges, LabVIEW, plain C).
//! All functions are `extern "C"`, pointer-in/pointer-out, no Rust types on
//! the boundary.

#![allow(unsafe_code)]

use crate::ring::SpscRing;
use crate::telemetry::Frame;
use std::ffi::c_void;
use std::slice;

/// Chi-squared `rᵀ diag(σ⁻²) r` over `n` lanes.
///
/// # Safety
/// `residual` and `inv_var` must point to `n` readable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn shbt_chi2(residual: *const f64, inv_var: *const f64, n: usize) -> f64 {
    let r = slice::from_raw_parts(residual, n);
    let iv = slice::from_raw_parts(inv_var, n);
    crate::rom::chi2(r, iv)
}

/// Dual-pump beat frequency `c·|λ₁⁻¹ − λ₂⁻¹|` [Hz].
#[no_mangle]
pub extern "C" fn shbt_beat_hz(lambda1_m: f64, lambda2_m: f64) -> f64 {
    shbt_rcwa_optics::material::beat_frequency(lambda1_m, lambda2_m)
}

/// DIN 2092 disc-spring force `P(s, E)` [N].
#[no_mangle]
pub extern "C" fn shbt_disc_force(
    de_m: f64,
    di_m: f64,
    t_m: f64,
    h0_m: f64,
    s_m: f64,
    youngs_pa: f64,
    poisson: f64,
) -> f64 {
    let d = shbt_fea_structural::DiscSpring {
        outer_diameter: de_m,
        inner_diameter: di_m,
        thickness: t_m,
        cone_height: h0_m,
    };
    d.force(
        s_m,
        shbt_fea_structural::material::Elastic {
            youngs: youngs_pa,
            poisson,
        },
    )
}

/// Opaque ring handle for C consumers.
#[no_mangle]
pub extern "C" fn shbt_ring_heap(capacity: usize) -> *mut c_void {
    Box::into_raw(Box::new(SpscRing::heap(capacity.next_power_of_two()))) as *mut c_void
}

/// Producer push; returns 1 on success, 0 when full (frame dropped by caller
/// policy — see `shbt_ring_dropped`).
///
/// # Safety
/// `ring` must come from [`shbt_ring_heap`]/[`shbt_ring_attach`].
#[no_mangle]
pub unsafe extern "C" fn shbt_ring_push(
    ring: *mut c_void,
    channel: u8,
    seq: u64,
    timestamp_ns: u64,
    values: *const f64,
    lanes: usize,
) -> i32 {
    let r = &*(ring as *const SpscRing);
    let (p, _) = r.split();
    let mut f = Frame::ZERO;
    f.channel = match channel {
        1 => crate::telemetry::Channel::ThermocoupleN,
        2 => crate::telemetry::Channel::FbgStrain,
        3 => crate::telemetry::Channel::CcdSpectrometer,
        _ => crate::telemetry::Channel::Other,
    };
    f.seq = seq;
    f.timestamp_ns = timestamp_ns;
    let n = lanes.min(crate::telemetry::FRAME_VALUES);
    f.lanes = n as u8;
    f.values[..n].copy_from_slice(slice::from_raw_parts(values, n));
    p.try_push(f) as i32
}

/// Consumer pop; returns 1 and writes the frame, or 0 when empty.
///
/// # Safety
/// `ring` must be a valid handle; `values` must hold ≥5 `f64`s.
#[no_mangle]
pub unsafe extern "C" fn shbt_ring_pop(
    ring: *mut c_void,
    channel: *mut u8,
    seq: *mut u64,
    timestamp_ns: *mut u64,
    values: *mut f64,
    lanes: *mut usize,
) -> i32 {
    let r = &*(ring as *const SpscRing);
    let (_, c) = r.split();
    match c.pop() {
        Some(f) => {
            *channel = f.channel as u8;
            *seq = f.seq;
            *timestamp_ns = f.timestamp_ns;
            let n = f.lanes as usize;
            *lanes = n;
            slice::from_raw_parts_mut(values, n).copy_from_slice(&f.values[..n]);
            1
        }
        None => 0,
    }
}

/// Frames dropped by the producer.
#[no_mangle]
pub unsafe extern "C" fn shbt_ring_dropped(ring: *mut c_void) -> usize {
    let r = &*(ring as *const SpscRing);
    let (_, c) = r.split();
    c.dropped()
}

/// Destroys a ring handle created by [`shbt_ring_heap`].
///
/// # Safety
/// `ring` must be a handle from `shbt_ring_heap` and not used afterwards.
#[no_mangle]
pub unsafe extern "C" fn shbt_ring_destroy(ring: *mut c_void) {
    drop(Box::from_raw(ring as *mut SpscRing));
}
