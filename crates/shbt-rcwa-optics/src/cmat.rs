//! Dense complex matrix helpers over `faer` used by the S-matrix recursion.

use faer::linalg::solvers::{DenseSolveCore, Solve};
use faer::prelude::*;

pub use faer::c64;

/// Dense column-major complex matrix.
pub type CMat = Mat<c64>;

/// Complex unit.
pub const I: c64 = c64 { re: 0.0, im: 1.0 };
/// Complex zero.
pub const ZERO: c64 = c64 { re: 0.0, im: 0.0 };
/// Complex one.
pub const ONE: c64 = c64 { re: 1.0, im: 0.0 };

/// `n × n` identity.
pub fn eye(n: usize) -> CMat {
    Mat::from_fn(n, n, |i, j| if i == j { ONE } else { ZERO })
}

/// Diagonal matrix from a slice.
pub fn diag(d: &[c64]) -> CMat {
    Mat::from_fn(d.len(), d.len(), |i, j| if i == j { d[i] } else { ZERO })
}

/// Real diagonal matrix.
pub fn diag_re(d: &[f64]) -> CMat {
    Mat::from_fn(d.len(), d.len(), |i, j| {
        if i == j {
            c64::new(d[i], 0.0)
        } else {
            ZERO
        }
    })
}

/// LU inverse.
pub fn inv(a: &CMat) -> CMat {
    a.partial_piv_lu().inverse()
}

/// Solve `A X = B`.
pub fn solve(a: &CMat, b: &CMat) -> CMat {
    a.partial_piv_lu().solve(b)
}

/// Block matrix `[[a, b], [c, d]]` of equally sized square blocks.
pub fn block2(a: &CMat, b: &CMat, c: &CMat, d: &CMat) -> CMat {
    let n = a.nrows();
    Mat::from_fn(2 * n, 2 * n, |i, j| match (i < n, j < n) {
        (true, true) => a[(i, j)],
        (true, false) => b[(i, j - n)],
        (false, true) => c[(i - n, j)],
        (false, false) => d[(i - n, j - n)],
    })
}

/// Vertical stack `[a; b]` of column vectors / matrices with equal columns.
pub fn vstack(a: &CMat, b: &CMat) -> CMat {
    let n = a.nrows();
    Mat::from_fn(n + b.nrows(), a.ncols(), |i, j| {
        if i < n {
            a[(i, j)]
        } else {
            b[(i - n, j)]
        }
    })
}

/// Upper / lower halves of a `2n × m` matrix.
pub fn split_rows(a: &CMat) -> (CMat, CMat) {
    let n = a.nrows() / 2;
    (
        Mat::from_fn(n, a.ncols(), |i, j| a[(i, j)]),
        Mat::from_fn(n, a.ncols(), |i, j| a[(i + n, j)]),
    )
}

/// Eigen-decomposition `A = W Λ W⁻¹` of a general complex matrix, returning
/// `(W, λ)` with `λ` the eigenvalues.
pub fn eig(a: &CMat) -> (CMat, Vec<c64>) {
    let e = a
        .eigen()
        .expect("RCWA mode matrix eigen-decomposition failed to converge");
    let s = e.S().column_vector();
    let lam: Vec<c64> = (0..s.nrows()).map(|i| s[i]).collect();
    (e.U().to_owned(), lam)
}

/// Max-abs (∞ entry) norm.
pub fn max_abs(a: &CMat) -> f64 {
    let mut m = 0.0f64;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            m = m.max(a[(i, j)].norm());
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eig_reconstructs_matrix() {
        let a = Mat::from_fn(4, 4, |i, j| {
            c64::new((i * 4 + j) as f64, (i as f64) - (j as f64))
        });
        let (w, lam) = eig(&a);
        let recon = &w * diag(&lam) * inv(&w);
        assert!(max_abs(&(recon - &a)) < 1e-11);
    }
}
