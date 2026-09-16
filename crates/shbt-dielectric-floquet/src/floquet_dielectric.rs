//! 512-bit Floquet dielectric tensor inversion (update-11.1 Task 1).
//!
//! Inverts the composite `(G, m)` dielectric matrix
//! `ε_{GG'}^{mn} = δ_{GG'}δ_{mn} − V_c(q+G) χ_{GG'}^{mn}(q, ω)` at 512-bit /
//! 440-mantissa-bit precision ([`Complex512`]) via Gauss–Jordan elimination
//! with partial pivoting.
//!
//! Mode truncation: `N_G = 25 × 25 = 625` spatial modes,
//! `N_F = 12 ⇒ N_T = 2N_F + 1 = 25` temporal harmonics, total dimension
//! `N = N_G · N_T = 15 625`. The flat index map is
//! `i = (m · N_G + G) · N_T + n`.

use shbt_core_math::precision::{free_float_cache, Complex512};

/// Spatial mode count `N_G = 25 × 25`.
pub const NUM_SPATIAL_MODES: usize = 625;
/// Temporal harmonic count `N_T = 2·12 + 1`.
pub const NUM_TEMPORAL_MODES: usize = 25;
/// Full composite matrix dimension `N = N_G · N_T`.
pub const TOTAL_DIM: usize = NUM_SPATIAL_MODES * NUM_TEMPORAL_MODES;

/// Gauss–Jordan inverter for the Floquet dielectric tensor at
/// [`PRECISION_BITS`](shbt_core_math::precision::PRECISION_BITS) mantissa
/// bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FloquetDielectricSolver {
    /// Spatial (G-vector) modes `N_G` — 625 for the production truncation.
    pub num_spatial_modes: usize,
    /// Temporal Floquet harmonics `N_T` — 25 for the production truncation.
    pub num_temporal_modes: usize,
    /// Composite dimension `N_G · N_T` — 15 625 for the production truncation.
    pub total_dim: usize,
}

impl FloquetDielectricSolver {
    /// Production truncation: 625 spatial × 25 temporal = 15 625.
    pub fn new() -> Self {
        Self::with_dimensions(NUM_SPATIAL_MODES, NUM_TEMPORAL_MODES)
    }

    /// Solver for an arbitrary `(N_G, N_T)` truncation.
    ///
    /// # Panics
    /// If `spatial * temporal` overflows `usize`.
    pub fn with_dimensions(num_spatial_modes: usize, num_temporal_modes: usize) -> Self {
        let total_dim = num_spatial_modes
            .checked_mul(num_temporal_modes)
            .expect("mode counts overflow usize");
        Self {
            num_spatial_modes,
            num_temporal_modes,
            total_dim,
        }
    }

    /// Inverts the `total_dim × total_dim` dielectric matrix `epsilon`
    /// (row-major, flat composite index) at 440-bit mantissa precision.
    ///
    /// Returns the inverse in the same layout, or
    /// `Err("Singular matrix encountered in Floquet solver")` when a pivot of
    /// strictly zero norm is found.
    ///
    /// # Panics
    /// If `epsilon.len() != total_dim²`.
    pub fn invert_dielectric_tensor(
        &self,
        epsilon: &[Complex512],
    ) -> Result<Vec<Complex512>, String> {
        assert_eq!(
            epsilon.len(),
            self.total_dim * self.total_dim,
            "Epsilon matrix dimension mismatch"
        );
        let n = self.total_dim;
        let mut a = epsilon.to_vec();
        let mut inv = vec![Complex512::zero(); n * n];
        for i in 0..n {
            inv[i * n + i] = Complex512::one();
        }

        for col in 0..n {
            // Partial pivot: row ≥ col with largest pivot norm.
            let mut pivot_row = col;
            let mut pivot_norm = a[col * n + col].norm();
            for row in (col + 1)..n {
                let cand = a[row * n + col].norm();
                if cand > pivot_norm {
                    pivot_norm = cand;
                    pivot_row = row;
                }
            }
            if pivot_norm == 0 {
                free_float_cache();
                return Err("Singular matrix encountered in Floquet solver".to_string());
            }
            if pivot_row != col {
                for k in 0..n {
                    a.swap(col * n + k, pivot_row * n + k);
                    inv.swap(col * n + k, pivot_row * n + k);
                }
            }

            // Normalize the pivot row by 1/pivot.
            let mut scale = Complex512::one();
            scale.div(&a[col * n + col].clone());
            for k in 0..n {
                let scaled = {
                    let mut t = a[col * n + k].clone();
                    t.mul(&scale);
                    t
                };
                a[col * n + k] = scaled;
                let scaled_inv = {
                    let mut t = inv[col * n + k].clone();
                    t.mul(&scale);
                    t
                };
                inv[col * n + k] = scaled_inv;
            }

            // Eliminate the column from every other row.
            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = a[row * n + col].clone();
                if factor.norm_sqr() == 0 {
                    continue;
                }
                for k in 0..n {
                    let mut t = a[col * n + k].clone();
                    t.mul(&factor);
                    a[row * n + k].sub(&t);
                    let mut t = inv[col * n + k].clone();
                    t.mul(&factor);
                    inv[row * n + k].sub(&t);
                }
            }
        }

        free_float_cache();
        Ok(inv)
    }
}

impl Default for FloquetDielectricSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_dimensions_are_15625() {
        let s = FloquetDielectricSolver::new();
        assert_eq!(s.num_spatial_modes, 625);
        assert_eq!(s.num_temporal_modes, 25);
        assert_eq!(s.total_dim, 15_625);
        assert_eq!(TOTAL_DIM, 15_625);
    }

    #[test]
    fn identity_inverts_to_identity() {
        let s = FloquetDielectricSolver::with_dimensions(1, 4);
        let n = s.total_dim;
        let mut eps = vec![Complex512::zero(); n * n];
        for i in 0..n {
            eps[i * n + i] = Complex512::one();
        }
        let inv = s.invert_dielectric_tensor(&eps).unwrap();
        for i in 0..n {
            for j in 0..n {
                let want = if i == j { 1.0 } else { 0.0 };
                assert_eq!(inv[i * n + j].re, want);
                assert_eq!(inv[i * n + j].im, 0.0);
            }
        }
    }

    #[test]
    fn diagonal_two_by_two_inverts() {
        let s = FloquetDielectricSolver::with_dimensions(1, 2);
        let eps = vec![
            Complex512::new(2.0, 0.0),
            Complex512::new(0.0, 0.0),
            Complex512::new(0.0, 0.0),
            Complex512::new(4.0, 0.0),
        ];
        let inv = s.invert_dielectric_tensor(&eps).unwrap();
        assert_eq!(inv[0].re, 0.5);
        assert_eq!(inv[3].re, 0.25);
    }

    #[test]
    fn complex_two_by_two_inverts() {
        // A = [[1+i, 2], [0, 1-i]]; A·A⁻¹ = I.
        let s = FloquetDielectricSolver::with_dimensions(1, 2);
        let eps = vec![
            Complex512::new(1.0, 1.0),
            Complex512::new(2.0, 0.0),
            Complex512::new(0.0, 0.0),
            Complex512::new(1.0, -1.0),
        ];
        let inv = s.invert_dielectric_tensor(&eps).unwrap();
        let tol = rug::Float::with_val(shbt_core_math::precision::PRECISION_BITS, 1e-100);
        // Verify A · A⁻¹ = I.
        let n = 2usize;
        for i in 0..n {
            for j in 0..n {
                let mut acc = Complex512::zero();
                for k in 0..n {
                    let mut t = eps[i * n + k].clone();
                    t.mul(&inv[k * n + j]);
                    acc.add(&t);
                }
                let want = if i == j { 1.0 } else { 0.0 };
                let diff =
                    rug::Float::with_val(shbt_core_math::precision::PRECISION_BITS, &acc.re - want);
                assert!(diff.abs() < tol, "re[{i},{j}] = {}", acc.re);
                assert!(acc.im.clone().abs() < tol, "im[{i},{j}] = {}", acc.im);
            }
        }
    }

    #[test]
    fn singular_matrix_returns_error() {
        let s = FloquetDielectricSolver::with_dimensions(1, 2);
        let eps = vec![Complex512::zero(); 4];
        let err = s.invert_dielectric_tensor(&eps).unwrap_err();
        assert_eq!(err, "Singular matrix encountered in Floquet solver");
    }

    #[test]
    #[should_panic(expected = "Epsilon matrix dimension mismatch")]
    fn dimension_mismatch_panics() {
        let s = FloquetDielectricSolver::with_dimensions(1, 2);
        let _ = s.invert_dielectric_tensor(&[Complex512::zero()]);
    }
}
