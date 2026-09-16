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

/// Mantissa bit depth of the 512-bit precision tier used by the Floquet
/// dielectric inversion (update-11.1 Task 1). A 512-bit register layout
/// carries 440 mantissa bits.
pub const PRECISION_BITS: u32 = 440;

/// 512-bit complex scalar backed by two [`rug::Float`]s at
/// [`PRECISION_BITS`] mantissa bits.
///
/// All arithmetic is in place so the 15 625² Floquet tensor does not
/// allocate per elementary operation.
#[derive(Clone, Debug)]
pub struct Complex512 {
    /// Real part.
    pub re: rug::Float,
    /// Imaginary part.
    pub im: rug::Float,
}

impl Complex512 {
    /// Creates a scalar from `f64` parts at [`PRECISION_BITS`].
    pub fn new(re: f64, im: f64) -> Self {
        assert_eq!(
            PRECISION_BITS, 440,
            "512-bit float requires 440 mantissa bits"
        );
        Self {
            re: rug::Float::with_val(PRECISION_BITS, re),
            im: rug::Float::with_val(PRECISION_BITS, im),
        }
    }

    /// `0 + 0i` at [`PRECISION_BITS`].
    pub fn zero() -> Self {
        Self::new(0.0, 0.0)
    }

    /// `1 + 0i` at [`PRECISION_BITS`].
    pub fn one() -> Self {
        Self::new(1.0, 0.0)
    }

    /// `self += rhs`.
    pub fn add(&mut self, rhs: &Self) {
        self.re += &rhs.re;
        self.im += &rhs.im;
    }

    /// `self -= rhs`.
    pub fn sub(&mut self, rhs: &Self) {
        self.re -= &rhs.re;
        self.im -= &rhs.im;
    }

    /// `self *= rhs` (complex product).
    pub fn mul(&mut self, rhs: &Self) {
        let re = rug::Float::with_val(PRECISION_BITS, &self.re * &rhs.re - &self.im * &rhs.im);
        let im = rug::Float::with_val(PRECISION_BITS, &self.re * &rhs.im + &self.im * &rhs.re);
        self.re = re;
        self.im = im;
    }

    /// `self /= rhs` (complex quotient, `rhs · conj(rhs)` real division).
    pub fn div(&mut self, rhs: &Self) {
        let denom = rug::Float::with_val(PRECISION_BITS, &rhs.re * &rhs.re + &rhs.im * &rhs.im);
        let re = rug::Float::with_val(PRECISION_BITS, &self.re * &rhs.re + &self.im * &rhs.im);
        let im = rug::Float::with_val(PRECISION_BITS, &self.im * &rhs.re - &self.re * &rhs.im);
        self.re = re / &denom;
        self.im = im / &denom;
    }

    /// `|self|²` as a real [`rug::Float`].
    pub fn norm_sqr(&self) -> rug::Float {
        rug::Float::with_val(PRECISION_BITS, &self.re * &self.re + &self.im * &self.im)
    }

    /// `|self|` (complex modulus) as a real [`rug::Float`].
    pub fn norm(&self) -> rug::Float {
        self.norm_sqr().sqrt()
    }
}

/// Releases MPFR's cached allocations. Call after a large arbitrary-precision
/// solve completes so the freed cache does not linger in the process.
///
/// MPFR's cache is process-global; solvers call this after a large inversion
/// completes and no other arbitrary-precision work is in flight.
pub fn free_float_cache() {
    rug::float::free_cache(rug::float::FreeCache::All);
}

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
    fn complex512_arithmetic() {
        assert_eq!(PRECISION_BITS, 440);
        let mut a = Complex512::new(1.5, -2.0);
        let b = Complex512::new(0.25, 3.0);
        a.add(&b);
        assert_eq!(a.re, 1.75);
        assert_eq!(a.im, 1.0);
        a.sub(&b);
        assert_eq!(a.re, 1.5);
        assert_eq!(a.im, -2.0);
        // (1.5 - 2i)(0.25 + 3i) = 6.375 + 4i
        a.mul(&b);
        assert_eq!(a.re, 6.375);
        assert_eq!(a.im, 4.0);
        a.div(&b);
        assert!(
            rug::Float::with_val(PRECISION_BITS, &a.re - 1.5).abs()
                < rug::Float::with_val(PRECISION_BITS, 1e-100)
        );
        assert!(
            rug::Float::with_val(PRECISION_BITS, &a.im + 2.0).abs()
                < rug::Float::with_val(PRECISION_BITS, 1e-100)
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
