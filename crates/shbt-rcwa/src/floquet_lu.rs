//! Allocation-free complex LU engine for Floquet reciprocal-space systems.
//!
//! Every routine operates on caller-supplied row-major buffers; nothing here
//! touches the heap. The only allocations live in the C-ABI / PyO3 wrappers,
//! which own their workspaces.

#![allow(unsafe_code)]

use std::ops::{Add, Div, Mul, Neg, Sub};

/// Plain `#[repr(C)]` complex scalar compatible with `double _Complex` layout.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Complex {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex {
    /// Constructs `re + i·im`.
    #[inline(always)]
    pub const fn new(re: f64, im: f64) -> Self {
        Complex { re, im }
    }

    /// `|z|²`.
    #[inline(always)]
    pub fn norm_sq(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Complex conjugate.
    #[inline(always)]
    pub fn conj(&self) -> Self {
        Complex {
            re: self.re,
            im: -self.im,
        }
    }
}

impl Add for Complex {
    type Output = Self;
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Complex {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

impl Sub for Complex {
    type Output = Self;
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        Complex {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}

impl Mul for Complex {
    type Output = Self;
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        Complex {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

impl Div for Complex {
    type Output = Self;
    #[inline(always)]
    fn div(self, other: Self) -> Self {
        let denom = other.re * other.re + other.im * other.im;
        Complex {
            re: (self.re * other.re + self.im * other.im) / denom,
            im: (self.im * other.re - self.re * other.im) / denom,
        }
    }
}

impl Neg for Complex {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Complex {
            re: -self.re,
            im: -self.im,
        }
    }
}

/// Pivot magnitude (squared) below which the matrix is treated as singular.
pub const SINGULAR_THRESHOLD: f64 = 1e-32;

/// In-place LU decomposition with partial (row) pivoting.
///
/// `a` is an `n × n` row-major matrix; on success it holds the unit-lower `L`
/// (strictly below the diagonal) and `U` (on and above) factors. `pivot[i]`
/// records the original row that ended up in row `i`.
pub fn lu_decompose(a: &mut [Complex], n: usize, pivot: &mut [usize]) -> Result<(), &'static str> {
    if a.len() < n * n {
        return Err("Matrix buffer size is smaller than n^2");
    }
    if pivot.len() < n {
        return Err("Pivot buffer size is smaller than n");
    }

    for (i, p) in pivot.iter_mut().enumerate().take(n) {
        *p = i;
    }

    for i in 0..n {
        let mut max_val = 0.0;
        let mut max_idx = i;
        for r in i..n {
            let val = a[r * n + i].norm_sq();
            if val > max_val {
                max_val = val;
                max_idx = r;
            }
        }

        if max_val < SINGULAR_THRESHOLD {
            return Err("Singular matrix encountered during LU decomposition");
        }

        if max_idx != i {
            for c in 0..n {
                a.swap(i * n + c, max_idx * n + c);
            }
            pivot.swap(i, max_idx);
        }

        let pivot_val = a[i * n + i];
        for r in (i + 1)..n {
            let factor = a[r * n + i] / pivot_val;
            a[r * n + i] = factor;
            for c in (i + 1)..n {
                let sub_val = factor * a[i * n + c];
                a[r * n + c] = a[r * n + c] - sub_val;
            }
        }
    }
    Ok(())
}

/// Solves `A x = b` given the factors produced by [`lu_decompose`].
pub fn lu_solve(
    lu: &[Complex],
    pivot: &[usize],
    b: &[Complex],
    n: usize,
    x: &mut [Complex],
) -> Result<(), &'static str> {
    if lu.len() < n * n || pivot.len() < n || b.len() < n || x.len() < n {
        return Err("Buffer size mismatch in linear solver");
    }

    for i in 0..n {
        x[i] = b[pivot[i]];
    }

    for i in 0..n {
        for j in 0..i {
            let sub = lu[i * n + j] * x[j];
            x[i] = x[i] - sub;
        }
    }

    for i in (0..n).rev() {
        for j in (i + 1)..n {
            let sub = lu[i * n + j] * x[j];
            x[i] = x[i] - sub;
        }
        x[i] = x[i] / lu[i * n + i];
    }
    Ok(())
}

/// Computes `A⁻¹` column by column from the LU factors into `inv` (row-major).
///
/// `col_workspace` and `solve_workspace` must each hold at least `n` entries.
pub fn lu_invert(
    lu: &[Complex],
    pivot: &[usize],
    n: usize,
    inv: &mut [Complex],
    col_workspace: &mut [Complex],
    solve_workspace: &mut [Complex],
) -> Result<(), &'static str> {
    if lu.len() < n * n
        || pivot.len() < n
        || inv.len() < n * n
        || col_workspace.len() < n
        || solve_workspace.len() < n
    {
        return Err("Workspace slice length is insufficient");
    }

    for col in 0..n {
        for entry in col_workspace.iter_mut().take(n) {
            *entry = Complex::new(0.0, 0.0);
        }
        col_workspace[col] = Complex::new(1.0, 0.0);

        lu_solve(lu, pivot, col_workspace, n, solve_workspace)?;

        for row in 0..n {
            inv[row * n + col] = solve_workspace[row];
        }
    }
    Ok(())
}

/// Solves the complex Floquet system `A x = b` through the C ABI.
///
/// Returns `0` on success, `-1` on a null pointer, `-2` if `A` is singular and
/// `-3` on a solve-stage buffer error.
///
/// # Safety
/// `a_re`/`a_im` must point to `n*n` readable `f64`s (row-major), `b_re`/`b_im`
/// to `n` readable `f64`s and `x_re`/`x_im` to `n` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn shbt_rcwa_solve_floquet(
    a_re: *const f64,
    a_im: *const f64,
    b_re: *const f64,
    b_im: *const f64,
    n: usize,
    x_re: *mut f64,
    x_im: *mut f64,
) -> i32 {
    if a_re.is_null()
        || a_im.is_null()
        || b_re.is_null()
        || b_im.is_null()
        || x_re.is_null()
        || x_im.is_null()
    {
        return -1;
    }

    let mut a = vec![Complex::default(); n * n];
    let mut b = vec![Complex::default(); n];
    let mut pivot = vec![0usize; n];
    let mut x = vec![Complex::default(); n];

    for (i, entry) in a.iter_mut().enumerate() {
        *entry = Complex::new(*a_re.add(i), *a_im.add(i));
    }
    for (i, entry) in b.iter_mut().enumerate() {
        *entry = Complex::new(*b_re.add(i), *b_im.add(i));
    }

    if lu_decompose(&mut a, n, &mut pivot).is_err() {
        return -2;
    }
    if lu_solve(&a, &pivot, &b, n, &mut x).is_err() {
        return -3;
    }

    for (i, value) in x.iter().enumerate() {
        *x_re.add(i) = value.re;
        *x_im.add(i) = value.im;
    }

    0
}

#[cfg(feature = "pyo3")]
mod python {
    use super::{lu_decompose, lu_solve, Complex};
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::prelude::*;

    /// Reusable LU workspace for `n × n` complex Floquet systems.
    #[pyclass]
    pub struct FloquetRCWASolver {
        n: usize,
        lu_matrix: Vec<Complex>,
        pivot: Vec<usize>,
    }

    #[pymethods]
    impl FloquetRCWASolver {
        #[new]
        fn new(n: usize) -> Self {
            FloquetRCWASolver {
                n,
                lu_matrix: vec![Complex::default(); n * n],
                pivot: vec![0; n],
            }
        }

        /// Solves `A x = b`; returns `(x_re, x_im)`.
        fn solve(
            &mut self,
            a_re: Vec<f64>,
            a_im: Vec<f64>,
            b_re: Vec<f64>,
            b_im: Vec<f64>,
        ) -> PyResult<(Vec<f64>, Vec<f64>)> {
            let n = self.n;
            if a_re.len() != n * n || a_im.len() != n * n || b_re.len() != n || b_im.len() != n {
                return Err(PyValueError::new_err(
                    "Input dimensions do not match constructor size",
                ));
            }
            for (slot, (re, im)) in self.lu_matrix.iter_mut().zip(a_re.iter().zip(&a_im)) {
                *slot = Complex::new(*re, *im);
            }
            lu_decompose(&mut self.lu_matrix, n, &mut self.pivot)
                .map_err(PyRuntimeError::new_err)?;

            let b: Vec<Complex> = b_re
                .iter()
                .zip(&b_im)
                .map(|(re, im)| Complex::new(*re, *im))
                .collect();
            let mut x = vec![Complex::default(); n];
            lu_solve(&self.lu_matrix, &self.pivot, &b, n, &mut x)
                .map_err(PyRuntimeError::new_err)?;

            Ok((
                x.iter().map(|z| z.re).collect(),
                x.iter().map(|z| z.im).collect(),
            ))
        }
    }

    #[pymodule]
    fn shbt_rcwa(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<FloquetRCWASolver>()?;
        Ok(())
    }
}

#[cfg(feature = "pyo3")]
pub use python::FloquetRCWASolver;

#[cfg(test)]
mod tests {
    use super::*;

    fn matmul(a: &[Complex], x: &[Complex], n: usize) -> Vec<Complex> {
        (0..n)
            .map(|i| (0..n).fold(Complex::default(), |acc, k| acc + a[i * n + k] * x[k]))
            .collect()
    }

    fn close(a: Complex, b: Complex) -> bool {
        (a - b).norm_sq().sqrt() < 1e-10
    }

    fn sample_matrix() -> Vec<Complex> {
        vec![
            Complex::new(4.0, 1.0),
            Complex::new(-2.0, 0.5),
            Complex::new(1.0, -1.0),
            Complex::new(0.0, 2.0),
            Complex::new(3.0, 0.0),
            Complex::new(-1.0, 1.0),
            Complex::new(2.0, -0.5),
            Complex::new(1.0, 1.0),
            Complex::new(5.0, -2.0),
        ]
    }

    #[test]
    fn complex_arithmetic() {
        let a = Complex::new(3.0, 4.0);
        let b = Complex::new(1.0, 2.0);
        assert!(close(a + b, Complex::new(4.0, 6.0)));
        assert!(close(a - b, Complex::new(2.0, 2.0)));
        assert!(close(a * b, Complex::new(-5.0, 10.0)));
        assert!(close(a / b, Complex::new(2.2, -0.4)));
        assert!(close(-a, Complex::new(-3.0, -4.0)));
        assert!(close(a.conj(), Complex::new(3.0, -4.0)));
        assert_eq!(a.norm_sq(), 25.0);
    }

    #[test]
    fn lu_solve_reproduces_rhs() {
        let n = 3;
        let a = sample_matrix();
        let b = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(-1.0, 2.0),
        ];
        let mut lu = a.clone();
        let mut pivot = [0usize; 3];
        lu_decompose(&mut lu, n, &mut pivot).unwrap();
        let mut x = [Complex::default(); 3];
        lu_solve(&lu, &pivot, &b, n, &mut x).unwrap();
        let ax = matmul(&a, &x, n);
        assert!(ax.iter().zip(&b).all(|(l, r)| close(*l, *r)));
    }

    #[test]
    fn lu_invert_gives_identity() {
        let n = 3;
        let a = sample_matrix();
        let mut lu = a.clone();
        let mut pivot = [0usize; 3];
        lu_decompose(&mut lu, n, &mut pivot).unwrap();
        let mut inv = [Complex::default(); 9];
        let mut col = [Complex::default(); 3];
        let mut work = [Complex::default(); 3];
        lu_invert(&lu, &pivot, n, &mut inv, &mut col, &mut work).unwrap();
        for i in 0..n {
            for j in 0..n {
                let entry = (0..n).fold(Complex::default(), |acc, k| {
                    acc + a[i * n + k] * inv[k * n + j]
                });
                let expected = if i == j {
                    Complex::new(1.0, 0.0)
                } else {
                    Complex::default()
                };
                assert!(close(entry, expected), "({i},{j}) = {entry:?}");
            }
        }
    }

    /// 4x4 Hilbert matrix scaled by (1 + 0.5i): the inverse is the known
    /// integer Hilbert inverse divided by the scalar.
    fn hilbert4_complex() -> (Vec<Complex>, Vec<Complex>) {
        let s = Complex::new(1.0, 0.5);
        let h4 = [
            1.0,
            1.0 / 2.0,
            1.0 / 3.0,
            1.0 / 4.0,
            1.0 / 2.0,
            1.0 / 3.0,
            1.0 / 4.0,
            1.0 / 5.0,
            1.0 / 3.0,
            1.0 / 4.0,
            1.0 / 5.0,
            1.0 / 6.0,
            1.0 / 4.0,
            1.0 / 5.0,
            1.0 / 6.0,
            1.0 / 7.0,
        ];
        let h4_inv = [
            16.0, -120.0, 240.0, -140.0, -120.0, 1200.0, -2700.0, 1680.0, 240.0, -2700.0, 6480.0,
            -4200.0, -140.0, 1680.0, -4200.0, 2800.0,
        ];
        let a: Vec<Complex> = h4.iter().map(|&v| Complex::new(v, 0.0) * s).collect();
        let inv: Vec<Complex> = h4_inv.iter().map(|&v| Complex::new(v, 0.0) / s).collect();
        (a, inv)
    }

    #[test]
    fn lu_invert_4x4_matches_hilbert_reference() {
        let n = 4;
        let (a, expected_inv) = hilbert4_complex();
        let mut lu = a.clone();
        let mut pivot = [0usize; 4];
        lu_decompose(&mut lu, n, &mut pivot).unwrap();

        let mut inv = [Complex::default(); 16];
        let mut col = [Complex::default(); 4];
        let mut work = [Complex::default(); 4];
        lu_invert(&lu, &pivot, n, &mut inv, &mut col, &mut work).unwrap();

        for i in 0..16 {
            let err = (inv[i] - expected_inv[i]).norm_sq().sqrt();
            assert!(
                err <= 1e-10 * expected_inv[i].norm_sq().sqrt().max(1.0),
                "index {i}: err {err}"
            );
        }
        for i in 0..n {
            for j in 0..n {
                let entry = (0..n).fold(Complex::default(), |acc, k| {
                    acc + a[i * n + k] * inv[k * n + j]
                });
                let expected = if i == j {
                    Complex::new(1.0, 0.0)
                } else {
                    Complex::default()
                };
                assert!((entry - expected).norm_sq().sqrt() < 1e-7);
            }
        }
    }

    #[test]
    fn lu_solve_3x3_and_4x4_residuals_are_machine_precision() {
        // 3x3 residual already covered; here an ill-conditioned 4x4 (Hilbert)
        // through the full decompose -> solve chain.
        let n = 4;
        let (a, _) = hilbert4_complex();
        let b = vec![
            Complex::new(1.0, -0.5),
            Complex::new(0.25, 0.75),
            Complex::new(-2.0, 0.0),
            Complex::new(0.5, 0.5),
        ];
        let mut lu = a.clone();
        let mut pivot = [0usize; 4];
        lu_decompose(&mut lu, n, &mut pivot).unwrap();
        let mut x = [Complex::default(); 4];
        lu_solve(&lu, &pivot, &b, n, &mut x).unwrap();
        let ax = matmul(&a, &x, n);
        for i in 0..n {
            assert!((ax[i] - b[i]).norm_sq().sqrt() < 1e-9);
        }
    }

    #[test]
    fn c_abi_solve_is_stable_across_repeated_calls() {
        // Repeated solves of the same ill-conditioned system must return
        // identical results to machine precision (deterministic pivoting).
        let n = 4;
        let (a, _) = hilbert4_complex();
        let a_re: Vec<f64> = a.iter().map(|z| z.re).collect();
        let a_im: Vec<f64> = a.iter().map(|z| z.im).collect();
        let b_re = [1.0, -0.5, 0.25, 2.0];
        let b_im = [0.5, 1.0, -0.25, -1.0];

        let mut first = vec![Complex::default(); n];
        for call in 0..1000 {
            let mut x_re = [0.0; 4];
            let mut x_im = [0.0; 4];
            let status = unsafe {
                shbt_rcwa_solve_floquet(
                    a_re.as_ptr(),
                    a_im.as_ptr(),
                    b_re.as_ptr(),
                    b_im.as_ptr(),
                    n,
                    x_re.as_mut_ptr(),
                    x_im.as_mut_ptr(),
                )
            };
            assert_eq!(status, 0);
            let x: Vec<Complex> = (0..n).map(|i| Complex::new(x_re[i], x_im[i])).collect();
            if call == 0 {
                first = x;
            } else {
                for i in 0..n {
                    assert!(close(x[i], first[i]), "call {call} diverged at {i}");
                }
            }
        }

        let ax = matmul(&a, &first, n);
        for i in 0..n {
            assert!((ax[i] - Complex::new(b_re[i], b_im[i])).norm_sq().sqrt() < 1e-9);
        }
    }

    #[test]
    fn singular_matrix_is_rejected() {
        let mut a = vec![Complex::new(1.0, 0.0); 4];
        let mut pivot = [0usize; 2];
        assert!(lu_decompose(&mut a, 2, &mut pivot).is_err());
    }

    #[test]
    fn c_abi_solve_matches_direct() {
        let n = 3;
        let a = sample_matrix();
        let a_re: Vec<f64> = a.iter().map(|z| z.re).collect();
        let a_im: Vec<f64> = a.iter().map(|z| z.im).collect();
        let b_re = [1.0, 0.0, -1.0];
        let b_im = [0.0, 1.0, 2.0];
        let mut x_re = [0.0; 3];
        let mut x_im = [0.0; 3];
        let status = unsafe {
            shbt_rcwa_solve_floquet(
                a_re.as_ptr(),
                a_im.as_ptr(),
                b_re.as_ptr(),
                b_im.as_ptr(),
                n,
                x_re.as_mut_ptr(),
                x_im.as_mut_ptr(),
            )
        };
        assert_eq!(status, 0);
        let x: Vec<Complex> = (0..n).map(|i| Complex::new(x_re[i], x_im[i])).collect();
        let ax = matmul(&a, &x, n);
        for i in 0..n {
            assert!(close(ax[i], Complex::new(b_re[i], b_im[i])));
        }
        let status = unsafe {
            shbt_rcwa_solve_floquet(
                std::ptr::null(),
                a_im.as_ptr(),
                b_re.as_ptr(),
                b_im.as_ptr(),
                n,
                x_re.as_mut_ptr(),
                x_im.as_mut_ptr(),
            )
        };
        assert_eq!(status, -1);
    }
}
