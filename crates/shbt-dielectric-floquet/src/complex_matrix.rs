//! Minimal dense complex matrix over `faer` for composite-index inversion.

use faer::linalg::solvers::DenseSolveCore;
use faer::prelude::*;
use num_complex::Complex64;
use std::ops::{Index, IndexMut, Sub};

/// Dense complex matrix (column-major `faer` storage).
#[derive(Clone, Debug)]
pub struct CMat {
    inner: Mat<Complex64>,
}

impl CMat {
    /// `n × m` zero matrix.
    pub fn zeros(n: usize, m: usize) -> Self {
        Self {
            inner: Mat::zeros(n, m),
        }
    }

    /// Max-abs (entry ∞) norm.
    pub fn max_abs(&self) -> f64 {
        let mut m = 0.0f64;
        for j in 0..self.inner.ncols() {
            for i in 0..self.inner.nrows() {
                m = m.max(self.inner[(i, j)].norm());
            }
        }
        m
    }

    /// Column-sum (matrix 1-) norm.
    fn norm_1(&self) -> f64 {
        let (nr, nc) = (self.inner.nrows(), self.inner.ncols());
        (0..nc)
            .map(|j| (0..nr).map(|i| self.inner[(i, j)].norm()).sum::<f64>())
            .fold(0.0, f64::max)
    }

    /// LU inverse together with the `‖A‖₁ ‖A⁻¹‖₁` condition estimate.
    pub fn inverse_with_condition(&self) -> (Self, f64) {
        let inv = Self {
            inner: self.inner.partial_piv_lu().inverse(),
        };
        (inv.clone(), self.norm_1() * inv.norm_1())
    }
}

impl Index<(usize, usize)> for CMat {
    type Output = Complex64;
    fn index(&self, (i, j): (usize, usize)) -> &Complex64 {
        &self.inner[(i, j)]
    }
}

impl IndexMut<(usize, usize)> for CMat {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Complex64 {
        &mut self.inner[(i, j)]
    }
}

impl Sub<&CMat> for &CMat {
    type Output = CMat;
    fn sub(self, rhs: &CMat) -> CMat {
        CMat {
            inner: &self.inner - &rhs.inner,
        }
    }
}
