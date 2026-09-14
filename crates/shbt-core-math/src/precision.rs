//! Precision-policy model shared by all domain solvers.
//!
//! The specification (§1, "Precision Configuration Models") defines three
//! execution tiers. Solvers declare which tier a subsystem runs in so that the
//! selection is explicit, auditable and testable rather than implied by the
//! concrete scalar type used in a given routine.

/// Numeric execution tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PrecisionMode {
    /// Strict IEEE 754 binary64 with fast-math disabled. Real-time HIL and
    /// dynamic FEA solves.
    StrictF64,
    /// 128-bit signed fixed point (Q64.64). Phase accumulation and spatial grid
    /// tracking.
    FixedQ64x64,
    /// Arbitrary-precision binary floating point (MPFR-backed) with the given
    /// mantissa width in bits. Floquet matrix inversion and RCWA S-matrix
    /// composition.
    Arbitrary {
        /// Mantissa width in bits, clamped to `[MIN_ARBITRARY_BITS, MAX_ARBITRARY_BITS]`.
        mantissa_bits: u32,
    },
}

/// Lower bound of the dynamic arbitrary-precision range.
pub const MIN_ARBITRARY_BITS: u32 = 128;
/// Upper bound of the dynamic arbitrary-precision range.
pub const MAX_ARBITRARY_BITS: u32 = 512;
/// Mantissa width of IEEE 754 binary64, including the implicit bit.
pub const F64_MANTISSA_BITS: u32 = 53;

impl PrecisionMode {
    /// Chooses a mantissa width for an arbitrary-precision solve from the
    /// estimated condition number of the operator. Roughly `log2(kappa)` bits
    /// are lost to conditioning; the result keeps at least 53 significant bits
    /// after that loss and stays inside the supported range.
    pub fn arbitrary_for_condition_number(kappa: f64) -> Self {
        let wanted = F64_MANTISSA_BITS.saturating_add(bits_lost_to_conditioning(kappa));
        Self::Arbitrary {
            mantissa_bits: wanted.clamp(MIN_ARBITRARY_BITS, MAX_ARBITRARY_BITS),
        }
    }

    /// Effective number of significant bits carried by this tier.
    pub const fn significant_bits(self) -> u32 {
        match self {
            Self::StrictF64 => F64_MANTISSA_BITS,
            Self::FixedQ64x64 => 128,
            Self::Arbitrary { mantissa_bits } => mantissa_bits,
        }
    }
}

/// Significant bits lost when solving a system with condition number `kappa`,
/// `ceil(log2 kappa)`. Non-finite or sub-unity `kappa` saturate to
/// `MAX_ARBITRARY_BITS` / `0` respectively.
pub fn bits_lost_to_conditioning(kappa: f64) -> u32 {
    if !kappa.is_finite() {
        MAX_ARBITRARY_BITS
    } else if kappa > 1.0 {
        ceil_u32(libm_log2(kappa))
    } else {
        0
    }
}

fn ceil_u32(x: f64) -> u32 {
    let truncated = x as u32;
    if (truncated as f64) < x {
        truncated + 1
    } else {
        truncated
    }
}

/// `log2` for `no_std` builds. Splits the IEEE 754 exponent from the mantissa
/// and evaluates the mantissa term with a short series; accurate to well under
/// one bit, which is all the tier selection needs.
fn libm_log2(x: f64) -> f64 {
    let bits = x.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32 - 1023;
    let mantissa = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    // ln(m) for m in [1, 2) via atanh series: ln(m) = 2 * atanh((m-1)/(m+1)).
    let y = (mantissa - 1.0) / (mantissa + 1.0);
    let y2 = y * y;
    let mut term = y;
    let mut sum = 0.0;
    let mut k = 1.0;
    for _ in 0..12 {
        sum += term / k;
        term *= y2;
        k += 2.0;
    }
    exponent as f64 + 2.0 * sum * core::f64::consts::LOG2_E
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn condition_number_scaling() {
        assert_eq!(
            PrecisionMode::arbitrary_for_condition_number(1.0),
            PrecisionMode::Arbitrary { mantissa_bits: 128 }
        );
        // 2^100 lost bits -> 153 wanted.
        let m = PrecisionMode::arbitrary_for_condition_number(2f64.powi(100));
        assert_eq!(m, PrecisionMode::Arbitrary { mantissa_bits: 153 });
        assert_eq!(
            PrecisionMode::arbitrary_for_condition_number(f64::INFINITY),
            PrecisionMode::Arbitrary { mantissa_bits: 512 }
        );
        assert_eq!(
            PrecisionMode::arbitrary_for_condition_number(1e300),
            PrecisionMode::Arbitrary { mantissa_bits: 512 }
        );
    }

    #[test]
    fn log2_accuracy() {
        for &(x, expect) in &[
            (1.0, 0.0),
            (2.0, 1.0),
            (8.0, 3.0),
            (1024.0, 10.0),
            (3.0, 1.584_962_500_721_156),
        ] {
            assert!((libm_log2(x) - expect).abs() < 1e-9, "log2({x})");
        }
    }
}
