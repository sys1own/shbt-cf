//! Arbitrary-precision dense linear algebra on top of `rug::Float` (MPFR).
//!
//! Used for the ill-conditioned inversions of the Floquet and RCWA solvers.
//! Precision is chosen dynamically: a system is solved at a starting mantissa
//! width, its 1-norm condition number `κ₁(A) = ‖A‖₁‖A⁻¹‖₁` is evaluated from the
//! computed inverse, and the solve is repeated at a wider mantissa whenever the
//! bits lost to conditioning would leave fewer than the requested number of
//! trustworthy bits in the result. Width is clamped to the 128–512-bit range of
//! [`crate::precision`].

use core::fmt;

use rug::float::Round;
use rug::ops::{AssignRound, CompleteRound};
use rug::Float;

use crate::precision::{
    bits_lost_to_conditioning, PrecisionMode, MAX_ARBITRARY_BITS, MIN_ARBITRARY_BITS,
};

/// Errors from arbitrary-precision linear algebra.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MpError {
    /// The matrix is singular to working precision.
    Singular,
    /// Matrix / vector shapes do not agree.
    ShapeMismatch,
}

impl fmt::Display for MpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Singular => write!(f, "matrix is singular to working precision"),
            Self::ShapeMismatch => write!(f, "matrix and vector shapes do not agree"),
        }
    }
}

impl std::error::Error for MpError {}

/// Square dense matrix of `rug::Float` entries, all at the same precision.
#[derive(Clone, Debug, PartialEq)]
pub struct MpMatrix {
    n: usize,
    prec: u32,
    data: Vec<Float>,
}

impl MpMatrix {
    /// Zero matrix of dimension `n` at `prec` mantissa bits.
    pub fn zeros(n: usize, prec: u32) -> Self {
        Self {
            n,
            prec,
            data: (0..n * n).map(|_| Float::with_val(prec, 0)).collect(),
        }
    }

    /// Identity matrix.
    pub fn identity(n: usize, prec: u32) -> Self {
        let mut m = Self::zeros(n, prec);
        for i in 0..n {
            m.data[i * n + i].assign_round(1, Round::Nearest);
        }
        m
    }

    /// Builds from a row-major `f64` slice of length `n²`. Every `f64` is
    /// exactly representable, so the conversion is lossless.
    pub fn from_f64_row_major(n: usize, prec: u32, rows: &[f64]) -> Result<Self, MpError> {
        if rows.len() != n * n {
            return Err(MpError::ShapeMismatch);
        }
        Ok(Self {
            n,
            prec,
            data: rows.iter().map(|&x| Float::with_val(prec, x)).collect(),
        })
    }

    /// Dimension.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Mantissa precision in bits.
    pub fn precision(&self) -> u32 {
        self.prec
    }

    /// Element accessor.
    pub fn get(&self, i: usize, j: usize) -> &Float {
        &self.data[i * self.n + j]
    }

    /// Mutable element accessor.
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut Float {
        &mut self.data[i * self.n + j]
    }

    /// Returns a copy with every entry re-rounded to `prec` bits.
    pub fn with_precision(&self, prec: u32) -> Self {
        Self {
            n: self.n,
            prec,
            data: self.data.iter().map(|x| Float::with_val(prec, x)).collect(),
        }
    }

    /// Induced 1-norm (maximum absolute column sum).
    pub fn norm_1(&self) -> Float {
        let mut best = Float::with_val(self.prec, 0);
        for j in 0..self.n {
            let mut col = Float::with_val(self.prec, 0);
            for i in 0..self.n {
                col += self.get(i, j).clone().abs();
            }
            if col > best {
                best = col;
            }
        }
        best
    }

    /// Matrix–vector product.
    pub fn mul_vec(&self, x: &[Float]) -> Result<Vec<Float>, MpError> {
        if x.len() != self.n {
            return Err(MpError::ShapeMismatch);
        }
        Ok((0..self.n)
            .map(|i| {
                let mut acc = Float::with_val(self.prec, 0);
                for (j, xj) in x.iter().enumerate() {
                    acc += (self.get(i, j) * xj).complete(self.prec);
                }
                acc
            })
            .collect())
    }

    /// Matrix product.
    pub fn mul(&self, rhs: &Self) -> Result<Self, MpError> {
        if rhs.n != self.n {
            return Err(MpError::ShapeMismatch);
        }
        let mut out = Self::zeros(self.n, self.prec);
        for i in 0..self.n {
            for j in 0..self.n {
                let mut acc = Float::with_val(self.prec, 0);
                for k in 0..self.n {
                    acc += (self.get(i, k) * rhs.get(k, j)).complete(self.prec);
                }
                *out.get_mut(i, j) = acc;
            }
        }
        Ok(out)
    }

    /// LU factorisation with partial pivoting (in place, Doolittle form).
    /// Returns the row permutation. Fails if a pivot underflows to zero.
    fn lu_in_place(&mut self) -> Result<Vec<usize>, MpError> {
        let n = self.n;
        let mut perm: Vec<usize> = (0..n).collect();
        for k in 0..n {
            let mut pivot = k;
            let mut pivot_abs = self.get(k, k).clone().abs();
            for i in (k + 1)..n {
                let cand = self.get(i, k).clone().abs();
                if cand > pivot_abs {
                    pivot_abs = cand;
                    pivot = i;
                }
            }
            if pivot_abs.is_zero() {
                return Err(MpError::Singular);
            }
            if pivot != k {
                for j in 0..n {
                    let (a, b) = (k * n + j, pivot * n + j);
                    self.data.swap(a, b);
                }
                perm.swap(k, pivot);
            }
            let pivot_val = self.get(k, k).clone();
            for i in (k + 1)..n {
                let factor = (self.get(i, k) / &pivot_val).complete(self.prec);
                *self.get_mut(i, k) = factor.clone();
                for j in (k + 1)..n {
                    let sub = (&factor * self.get(k, j)).complete(self.prec);
                    *self.get_mut(i, j) -= sub;
                }
            }
        }
        Ok(perm)
    }

    /// Solves `A x = b` for a single right-hand side.
    pub fn solve(&self, b: &[Float]) -> Result<Vec<Float>, MpError> {
        if b.len() != self.n {
            return Err(MpError::ShapeMismatch);
        }
        let mut lu = self.clone();
        let perm = lu.lu_in_place()?;
        Ok(lu.lu_solve_permuted(&perm, b))
    }

    fn lu_solve_permuted(&self, perm: &[usize], b: &[Float]) -> Vec<Float> {
        let n = self.n;
        let mut y: Vec<Float> = perm
            .iter()
            .map(|&p| Float::with_val(self.prec, &b[p]))
            .collect();
        for i in 0..n {
            for j in 0..i {
                let sub = (self.get(i, j) * &y[j]).complete(self.prec);
                y[i] -= sub;
            }
        }
        for i in (0..n).rev() {
            for j in (i + 1)..n {
                let sub = (self.get(i, j) * &y[j]).complete(self.prec);
                y[i] -= sub;
            }
            let pivot = self.get(i, i).clone();
            y[i] /= pivot;
        }
        y
    }

    /// Matrix inverse via LU.
    pub fn inverse(&self) -> Result<Self, MpError> {
        let n = self.n;
        let mut lu = self.clone();
        let perm = lu.lu_in_place()?;
        let mut inv = Self::zeros(n, self.prec);
        for j in 0..n {
            let mut e: Vec<Float> = (0..n).map(|_| Float::with_val(self.prec, 0)).collect();
            e[j].assign_round(1, Round::Nearest);
            let col = lu.lu_solve_permuted(&perm, &e);
            for (i, v) in col.into_iter().enumerate() {
                *inv.get_mut(i, j) = v;
            }
        }
        Ok(inv)
    }

    /// 1-norm condition number `κ₁(A) = ‖A‖₁ ‖A⁻¹‖₁`, or `+∞` if singular.
    pub fn condition_number_1(&self) -> f64 {
        match self.inverse() {
            Ok(inv) => (self.norm_1() * inv.norm_1()).to_f64(),
            Err(_) => f64::INFINITY,
        }
    }
}

/// Result of an adaptive-precision solve.
#[derive(Clone, Debug)]
pub struct AdaptiveSolution {
    /// Solution vector at the final working precision.
    pub x: Vec<Float>,
    /// Working precision that produced `x`.
    pub precision: PrecisionMode,
    /// Estimated 1-norm condition number of the matrix.
    pub condition_number: f64,
    /// Whether the target accuracy was met within the 512-bit ceiling.
    pub converged: bool,
}

/// Solves `A x = b`, escalating mantissa width until at least `target_bits`
/// significant bits survive the conditioning loss `log₂ κ₁(A)`, or the 512-bit
/// ceiling is reached. `a` and `b` may be at any precision; they are re-rounded
/// to each working precision (lossless when they originate from `f64`).
pub fn solve_adaptive(
    a: &MpMatrix,
    b: &[Float],
    target_bits: u32,
) -> Result<AdaptiveSolution, MpError> {
    if b.len() != a.dim() {
        return Err(MpError::ShapeMismatch);
    }
    let mut prec = MIN_ARBITRARY_BITS;
    loop {
        let a_p = a.with_precision(prec);
        let b_p: Vec<Float> = b.iter().map(|v| Float::with_val(prec, v)).collect();
        let inv = a_p.inverse()?;
        let kappa = (a_p.norm_1() * inv.norm_1()).to_f64();
        let x = inv.mul_vec(&b_p)?;
        let required = bits_lost_to_conditioning(kappa).saturating_add(target_bits);
        let converged = prec >= required;
        if converged || prec >= MAX_ARBITRARY_BITS {
            return Ok(AdaptiveSolution {
                x,
                precision: PrecisionMode::Arbitrary {
                    mantissa_bits: prec,
                },
                condition_number: kappa,
                converged,
            });
        }
        prec = required.clamp(prec + 1, MAX_ARBITRARY_BITS);
        // Widen in whole 64-bit limbs so MPFR does not pay for odd sizes.
        prec = prec.div_ceil(64) * 64;
        prec = prec.min(MAX_ARBITRARY_BITS);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hilbert(n: usize) -> Vec<f64> {
        let mut v = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                v.push(1.0 / ((i + j + 1) as f64));
            }
        }
        v
    }

    #[test]
    fn identity_inverse_and_solve() {
        let a =
            MpMatrix::from_f64_row_major(3, 128, &[2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 8.0])
                .unwrap();
        let inv = a.inverse().unwrap();
        assert_eq!(inv.get(0, 0).to_f64(), 0.5);
        assert_eq!(inv.get(2, 2).to_f64(), 0.125);
        let b: Vec<Float> = [2.0, 4.0, 8.0]
            .iter()
            .map(|&v| Float::with_val(128, v))
            .collect();
        let x = a.solve(&b).unwrap();
        assert!(x.iter().all(|v| v.to_f64() == 1.0));
        assert_eq!(a.condition_number_1(), 4.0);
    }

    #[test]
    fn singular_is_detected() {
        let a = MpMatrix::from_f64_row_major(2, 128, &[1.0, 2.0, 2.0, 4.0]).unwrap();
        assert_eq!(a.inverse(), Err(MpError::Singular));
        assert_eq!(a.condition_number_1(), f64::INFINITY);
    }

    #[test]
    fn hilbert_needs_escalation() {
        // κ₁(H₁₂) ≈ 4e16 (> 2^54): 128 bits leaves ~74 trustworthy bits, so a
        // target of 100 bits must force escalation to 192.
        let n = 12;
        let a = MpMatrix::from_f64_row_major(n, 128, &hilbert(n)).unwrap();
        // b = H · 1 so the exact solution is the all-ones vector.
        let ones: Vec<Float> = (0..n).map(|_| Float::with_val(128, 1)).collect();
        let b = a.mul_vec(&ones).unwrap();
        let sol = solve_adaptive(&a, &b, 100).unwrap();
        assert!(sol.converged);
        assert!(sol.condition_number > 1e15, "{}", sol.condition_number);
        assert_eq!(
            sol.precision,
            PrecisionMode::Arbitrary { mantissa_bits: 192 }
        );
        for v in &sol.x {
            assert!((v.to_f64() - 1.0).abs() < 1e-12, "{v}");
        }
    }

    #[test]
    fn ceiling_is_respected() {
        // κ₁(H₂₀) ≈ 8e18 (> 2^62) → 63+450 bits demanded → clamps at 512, not converged.
        let n = 20;
        let a = MpMatrix::from_f64_row_major(n, 128, &hilbert(n)).unwrap();
        let b: Vec<Float> = (0..n).map(|_| Float::with_val(128, 1)).collect();
        let sol = solve_adaptive(&a, &b, 450).unwrap();
        assert!(sol.condition_number > 1e18, "{}", sol.condition_number);
        assert_eq!(
            sol.precision,
            PrecisionMode::Arbitrary { mantissa_bits: 512 }
        );
        assert!(!sol.converged);
    }
}
