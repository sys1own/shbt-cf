//! Q64.64 signed fixed-point arithmetic.
//!
//! The representation is a two's-complement `i128` whose lower 64 bits are the
//! fractional part. Addition and subtraction are exact (no exponent shifting),
//! so accumulators built on this type exhibit zero rounding drift over any
//! number of steps until the ±2^63 integer range is exceeded, at which point
//! the operation reports overflow instead of silently wrapping.

use core::ops::{Add, Neg, Sub};

/// Number of fractional bits in the representation.
pub const FRAC_BITS: u32 = 64;

/// Absolute resolution of one unit in the last place, `2^-64`.
pub const RESOLUTION: f64 = 5.421_010_862_427_522e-20;

/// Q64.64 fixed-point value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct Q64x64(i128);

impl Q64x64 {
    /// Zero.
    pub const ZERO: Self = Self(0);
    /// One.
    pub const ONE: Self = Self(1_i128 << FRAC_BITS);
    /// Largest representable value.
    pub const MAX: Self = Self(i128::MAX);
    /// Smallest representable value.
    pub const MIN: Self = Self(i128::MIN);

    /// Builds a value from its raw two's-complement bit pattern.
    #[inline]
    pub const fn from_bits(bits: i128) -> Self {
        Self(bits)
    }

    /// Returns the raw two's-complement bit pattern.
    #[inline]
    pub const fn to_bits(self) -> i128 {
        self.0
    }

    /// Builds a value from an integer, with no fractional part.
    #[inline]
    pub const fn from_int(value: i64) -> Self {
        Self((value as i128) << FRAC_BITS)
    }

    /// Converts a finite `f64` to fixed point, truncating toward zero below
    /// the 2^-64 resolution. Returns `None` for NaN, infinities and values
    /// outside the ±2^63 integer range.
    pub fn from_f64(value: f64) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }
        const TWO_POW_64: f64 = 18_446_744_073_709_551_616.0;
        const TWO_POW_127: f64 = 1.701_411_834_604_692_3e38;
        let scaled = value * TWO_POW_64;
        if !(-TWO_POW_127..TWO_POW_127).contains(&scaled) {
            return None;
        }
        Some(Self(scaled as i128))
    }

    /// Converts to the nearest `f64`. Lossy for values needing more than 53
    /// significant bits.
    #[inline]
    pub fn to_f64(self) -> f64 {
        (self.0 as f64) * RESOLUTION
    }

    /// Checked addition; `None` on overflow.
    #[inline]
    pub const fn checked_add(self, rhs: Self) -> Option<Self> {
        match self.0.checked_add(rhs.0) {
            Some(v) => Some(Self(v)),
            None => None,
        }
    }

    /// Checked subtraction; `None` on overflow.
    #[inline]
    pub const fn checked_sub(self, rhs: Self) -> Option<Self> {
        match self.0.checked_sub(rhs.0) {
            Some(v) => Some(Self(v)),
            None => None,
        }
    }

    /// Fixed-point multiplication with the intermediate carried at 256 bits so
    /// the result is correctly truncated toward negative infinity. Returns
    /// `None` if the product leaves the representable range.
    pub fn checked_mul(self, rhs: Self) -> Option<Self> {
        let (a_neg, a) = split_sign(self.0);
        let (b_neg, b) = split_sign(rhs.0);
        let (hi, lo) = mul_u128(a, b);
        // Shift the 256-bit product right by FRAC_BITS.
        let result_hi = hi >> FRAC_BITS;
        let result_lo = (hi << FRAC_BITS) | (lo >> FRAC_BITS);
        if result_hi != 0 {
            return None;
        }
        let magnitude = result_lo;
        if a_neg != b_neg {
            if magnitude > (1_u128 << 127) {
                return None;
            }
            Some(Self((magnitude as i128).wrapping_neg()))
        } else {
            if magnitude > i128::MAX as u128 {
                return None;
            }
            Some(Self(magnitude as i128))
        }
    }

    /// Integer part, rounded toward negative infinity.
    #[inline]
    pub const fn floor_int(self) -> i64 {
        (self.0 >> FRAC_BITS) as i64
    }

    /// Fractional part in `[0, 1)` as raw 64-bit fraction.
    #[inline]
    pub const fn frac_bits(self) -> u64 {
        self.0 as u64
    }
}

#[inline]
const fn split_sign(v: i128) -> (bool, u128) {
    if v < 0 {
        (true, v.unsigned_abs())
    } else {
        (false, v as u128)
    }
}

/// Full 128x128 -> 256-bit unsigned multiply, returned as (hi, lo).
fn mul_u128(a: u128, b: u128) -> (u128, u128) {
    const MASK: u128 = u64::MAX as u128;
    let (a_lo, a_hi) = (a & MASK, a >> 64);
    let (b_lo, b_hi) = (b & MASK, b >> 64);

    let ll = a_lo * b_lo;
    let lh = a_lo * b_hi;
    let hl = a_hi * b_lo;
    let hh = a_hi * b_hi;

    // Each term is < 2^64, so the sum of three fits in u128 without overflow.
    let mid = (ll >> 64) + (lh & MASK) + (hl & MASK);
    let lo = (ll & MASK) | (mid << 64);
    let hi = hh + (lh >> 64) + (hl >> 64) + (mid >> 64);
    (hi, lo)
}

impl Add for Q64x64 {
    type Output = Self;
    /// Panics on overflow in debug builds; use [`Q64x64::checked_add`] in hot loops.
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Q64x64 {
    type Output = Self;
    /// Panics on overflow in debug builds; use [`Q64x64::checked_sub`] in hot loops.
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Neg for Q64x64 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_round_trip() {
        assert_eq!(Q64x64::from_int(42).floor_int(), 42);
        assert_eq!(Q64x64::from_int(-42).floor_int(), -42);
        assert_eq!(Q64x64::from_int(1), Q64x64::ONE);
    }

    #[test]
    fn f64_round_trip() {
        let x = Q64x64::from_f64(0.5).unwrap();
        assert_eq!(x.to_bits(), 1_i128 << 63);
        assert_eq!(x.to_f64(), 0.5);
        assert!(Q64x64::from_f64(f64::NAN).is_none());
        assert!(Q64x64::from_f64(f64::INFINITY).is_none());
        assert!(Q64x64::from_f64(1e30).is_none());
    }

    #[test]
    fn accumulation_is_exact() {
        // 0.1 is not exactly representable in binary, but whatever bit pattern
        // it truncates to must accumulate exactly with no rounding drift.
        let step = Q64x64::from_f64(0.1).unwrap();
        let mut acc = Q64x64::ZERO;
        for _ in 0..1_000_000 {
            acc = acc + step;
        }
        assert_eq!(acc.to_bits(), step.to_bits() * 1_000_000);
    }

    #[test]
    fn multiplication() {
        let a = Q64x64::from_f64(1.5).unwrap();
        let b = Q64x64::from_f64(-2.0).unwrap();
        assert_eq!(a.checked_mul(b).unwrap().to_f64(), -3.0);
        assert_eq!(Q64x64::ONE.checked_mul(Q64x64::ONE), Some(Q64x64::ONE));
        assert!(Q64x64::MAX.checked_mul(Q64x64::from_int(2)).is_none());
    }

    #[test]
    fn overflow_is_reported() {
        assert!(Q64x64::MAX.checked_add(Q64x64::ONE).is_none());
        assert!(Q64x64::MIN.checked_sub(Q64x64::ONE).is_none());
    }
}
