//! Cross-platform bitwise regression suite (simulator_spec.pdf §6).
//!
//! Exercises the deterministic double-precision kernels and folds every result
//! into a single 64-bit digest compared bit-for-bit across x86-64 (AVX-512)
//! and ARM64 (NEON) CI runners. All exercised paths are restricted to IEEE-754
//! `+ − × ÷` and integer ops — no libm transcendentals, which are not
//! guaranteed bit-identical across platforms — so the digest is portable.

use shbt_cf_native::rom::chi2;
use shbt_cf_native::telemetry::{Channel, Frame};
use shbt_core_math::fixed::Q64x64;
use shbt_core_math::symplectic::{PhaseState, SeparableHamiltonian, Yoshida6};
use shbt_metrology_gum::engine::models;
use shbt_metrology_gum::random::Rng;

#[inline]
fn fold(acc: u64, bits: u64) -> u64 {
    // Deterministic non-commutative mix; order-dependent by design.
    acc.rotate_left(5)
        .wrapping_add(bits ^ 0x9e37_79b9_7f4a_7c15)
}

/// `H = p²/2 + q²/2 + q⁴/4`: quartic oscillator, libm-free.
struct Quartic;

impl SeparableHamiltonian<1> for Quartic {
    fn velocity(&self, p: &[f64; 1]) -> [f64; 1] {
        [p[0]]
    }
    fn potential_gradient(&self, q: &[f64; 1]) -> [f64; 1] {
        [q[0] + q[0] * q[0] * q[0]]
    }
    fn energy(&self, q: &[f64; 1], p: &[f64; 1]) -> f64 {
        0.5 * p[0] * p[0] + 0.5 * q[0] * q[0] + 0.25 * q[0].powi(4)
    }
}

fn kernel_digest() -> u64 {
    let mut acc = 0x5348_4254_3030_3031u64; // "SHBT0001"

    // 1. Xoshiro256** stream (integer ops only).
    let mut rng = Rng::new(0x5EED_2024);
    for _ in 0..4096 {
        acc = fold(acc, rng.next_u64());
    }

    // 2. Yoshida-6 symplectic integration, compensated accumulation.
    let mut s = PhaseState::new([1.25], [0.0]);
    Yoshida6::integrate(&Quartic, &mut s, 0.0078125, 100_000);
    for v in [s.q[0], s.p[0], s.q_lo[0], s.p_lo[0]] {
        acc = fold(acc, v.to_bits());
    }

    // 3. Q64.64 fixed-point accumulation (i128 arithmetic).
    let mut q = Q64x64::ZERO;
    let step = Q64x64::from_f64(0.000_976_562_5).unwrap(); // 2^-10 exact
    for _ in 0..8192 {
        q = q.checked_add(step).unwrap();
        q = q
            .checked_mul(Q64x64::from_f64(1.000_488_281_25).unwrap())
            .unwrap();
    }
    acc = fold(acc, q.to_bits() as u64);
    acc = fold(acc, (q.to_bits() >> 64) as u64);

    // 4. χ² residual and calorimetric measurand (mul-add only).
    let r = [0.5, -1.25, 3.125, 0.0625];
    let iv = [4.0, 16.0, 0.64, 2.56];
    acc = fold(acc, chi2(&r, &iv).to_bits());
    acc = fold(acc, models::power_net(&[45.0, 13.25]).to_bits());

    // 5. Frame layout digest — repr(C) bit pattern is platform-stable.
    let f = Frame::scalar(Channel::ThermocoupleN, 7, 1_234_567, 295.15);
    for v in f.values {
        acc = fold(acc, v.to_bits());
    }
    acc = fold(
        acc,
        f.seq ^ f.timestamp_ns.rotate_left(31) ^ f.channel as u64,
    );

    acc
}

#[test]
fn deterministic_kernels_bit_identical() {
    let digest = kernel_digest();
    // Golden digest generated on x86-64; the ARM64 CI job runs the same test
    // and must reproduce it exactly (spec §6 gate).
    assert_eq!(
        digest, GOLDEN_DIGEST,
        "bitwise regression mismatch: {digest:#018x}"
    );
    // Repeat: the kernel itself must be run-to-run deterministic.
    assert_eq!(digest, kernel_digest());
}

const GOLDEN_DIGEST: u64 = 0x03de_00d3_acde_d967;
