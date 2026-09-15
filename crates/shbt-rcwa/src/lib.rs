//! Dense crossed-grating RCWA primitives.
#![allow(missing_docs)]

use ndarray::{Array2, Array3, Axis};
use num_complex::Complex64;
use rayon::prelude::*;

pub const DEFAULT_HARMONICS: usize = 25;

#[derive(Debug, Clone, Copy)]
pub struct RcwaSystemSolver {
    pub spatial_harmonics_m: usize,
    pub spatial_harmonics_n: usize,
    pub free_space_wavelength: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LayerSMatrix {
    pub s11: Array2<Complex64>,
    pub s12: Array2<Complex64>,
    pub s21: Array2<Complex64>,
    pub s22: Array2<Complex64>,
}

pub trait BoundaryDiffractionSolver {
    fn compute_normal_vector_field(&self, roughness: &Array2<f64>) -> Array3<f64>;
    fn apply_li_factorization(
        &self,
        permittivity_tensor: &Array3<Complex64>,
        normal_vectors: &Array3<f64>,
    ) -> Array2<Complex64>;
    fn star_product_cascade(&self, layer_a: &LayerSMatrix, layer_b: &LayerSMatrix) -> LayerSMatrix;
}

impl RcwaSystemSolver {
    pub fn new(free_space_wavelength: f64) -> Self {
        Self {
            spatial_harmonics_m: DEFAULT_HARMONICS,
            spatial_harmonics_n: DEFAULT_HARMONICS,
            free_space_wavelength,
        }
    }

    pub fn with_harmonics(m: usize, n: usize, free_space_wavelength: f64) -> Self {
        assert!(m > 0 && n > 0);
        Self {
            spatial_harmonics_m: m,
            spatial_harmonics_n: n,
            free_space_wavelength,
        }
    }

    pub fn mode_count(&self) -> usize {
        self.spatial_harmonics_m * self.spatial_harmonics_n
    }

    pub fn identity_s_matrix(&self) -> LayerSMatrix {
        let size = 2 * self.mode_count();
        LayerSMatrix {
            s11: eye(size),
            s12: Array2::zeros((size, size)),
            s21: Array2::zeros((size, size)),
            s22: eye(size),
        }
    }

    /// Cascades layers from the incident side toward the transmitted side.
    pub fn solve_layers(&self, layers: &[LayerSMatrix]) -> LayerSMatrix {
        layers
            .iter()
            .fold(self.identity_s_matrix(), |global, layer| {
                self.star_product_cascade(&global, layer)
            })
    }
}

impl BoundaryDiffractionSolver for RcwaSystemSolver {
    fn compute_normal_vector_field(&self, roughness: &Array2<f64>) -> Array3<f64> {
        let (rows, cols) = roughness.dim();
        let mut normals = Array3::zeros((rows, cols, 3));
        for i in 0..rows {
            for j in 0..cols {
                let dx = if rows < 2 {
                    0.0
                } else if i == 0 {
                    roughness[[1, j]] - roughness[[0, j]]
                } else if i + 1 == rows {
                    roughness[[i, j]] - roughness[[i - 1, j]]
                } else {
                    0.5 * (roughness[[i + 1, j]] - roughness[[i - 1, j]])
                };
                let dy = if cols < 2 {
                    0.0
                } else if j == 0 {
                    roughness[[i, 1]] - roughness[[i, 0]]
                } else if j + 1 == cols {
                    roughness[[i, j]] - roughness[[i, j - 1]]
                } else {
                    0.5 * (roughness[[i, j + 1]] - roughness[[i, j - 1]])
                };
                let norm = (1.0 + dx * dx + dy * dy).sqrt();
                normals[[i, j, 0]] = -dx / norm;
                normals[[i, j, 1]] = -dy / norm;
                normals[[i, j, 2]] = 1.0 / norm;
            }
        }
        normals
    }

    /// Applies Li's matrix FFF locally and returns the transverse block matrix.
    /// The tensor layout is `(x, y, 9)`, with the final axis in row-major 3x3 order.
    fn apply_li_factorization(
        &self,
        epsilon: &Array3<Complex64>,
        normals: &Array3<f64>,
    ) -> Array2<Complex64> {
        assert_eq!(
            (epsilon.dim().0, epsilon.dim().1),
            (normals.dim().0, normals.dim().1),
            "epsilon and normals must share the spatial grid"
        );
        assert_eq!(epsilon.dim().2, 9);
        let samples = epsilon.len_of(Axis(0)) * epsilon.len_of(Axis(1));
        let blocks: Vec<[[Complex64; 2]; 2]> = (0..samples)
            .into_par_iter()
            .map(|flat| {
                let (x, y) = (
                    flat / epsilon.len_of(Axis(1)),
                    flat % epsilon.len_of(Axis(1)),
                );
                let nx = normals[[x, y, 0]];
                let ny = normals[[x, y, 1]];
                let e = |i: usize, j: usize| epsilon[[x, y, 3 * i + j]];
                let den = e(0, 0) * nx * nx + (e(0, 1) + e(1, 0)) * nx * ny + e(1, 1) * ny * ny;
                let safe_den = if den.norm() < 1e-14 {
                    Complex64::new(1e-14, 0.0)
                } else {
                    den
                };
                let qx = e(0, 0) * nx + e(0, 1) * ny;
                let qy = e(1, 0) * nx + e(1, 1) * ny;
                [
                    [
                        e(0, 0) - qx * nx + qx * nx / safe_den,
                        e(0, 1) - qx * ny + qx * ny / safe_den,
                    ],
                    [
                        e(1, 0) - qy * nx + qy * nx / safe_den,
                        e(1, 1) - qy * ny + qy * ny / safe_den,
                    ],
                ]
            })
            .collect();
        let mut result = Array2::zeros((2 * samples, 2 * samples));
        for (index, block) in blocks.iter().enumerate() {
            result[[index, index]] = block[0][0];
            result[[index, samples + index]] = block[0][1];
            result[[samples + index, index]] = block[1][0];
            result[[samples + index, samples + index]] = block[1][1];
        }
        result
    }

    fn star_product_cascade(&self, a: &LayerSMatrix, b: &LayerSMatrix) -> LayerSMatrix {
        let n = a.s11.nrows();
        assert!(
            a.s11.dim() == (n, n)
                && [
                    a.s12.dim(),
                    a.s21.dim(),
                    a.s22.dim(),
                    b.s11.dim(),
                    b.s12.dim(),
                    b.s21.dim(),
                    b.s22.dim()
                ]
                .iter()
                .all(|&d| d == (n, n))
        );
        let identity = eye(n);
        let left = inverse(&(&identity - &a.s12.dot(&b.s21)));
        let right = inverse(&(&identity - &b.s21.dot(&a.s12)));
        LayerSMatrix {
            s11: b.s11.dot(&left).dot(&a.s11),
            s12: &b.s12 + &b.s11.dot(&left).dot(&a.s12).dot(&b.s22),
            s21: &a.s21 + &a.s22.dot(&right).dot(&b.s21).dot(&a.s11),
            s22: a.s22.dot(&right).dot(&b.s22),
        }
    }
}

fn eye(size: usize) -> Array2<Complex64> {
    Array2::from_shape_fn((size, size), |(i, j)| {
        if i == j {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    })
}

fn inverse(matrix: &Array2<Complex64>) -> Array2<Complex64> {
    let size = matrix.nrows();
    assert_eq!(matrix.ncols(), size);
    let mut augmented = Array2::zeros((size, 2 * size));
    for row in 0..size {
        for column in 0..size {
            augmented[[row, column]] = matrix[[row, column]];
        }
        augmented[[row, size + row]] = Complex64::new(1.0, 0.0);
    }
    for pivot in 0..size {
        let pivot_row = (pivot..size)
            .max_by(|&left, &right| {
                augmented[[left, pivot]]
                    .norm()
                    .total_cmp(&augmented[[right, pivot]].norm())
            })
            .unwrap();
        assert!(
            augmented[[pivot_row, pivot]].norm() > 1e-14,
            "singular RCWA star-product solve"
        );
        if pivot_row != pivot {
            for column in 0..2 * size {
                augmented.swap((pivot, column), (pivot_row, column));
            }
        }
        let pivot_value = augmented[[pivot, pivot]];
        for column in 0..2 * size {
            augmented[[pivot, column]] /= pivot_value;
        }
        for row in 0..size {
            if row == pivot {
                continue;
            }
            let factor = augmented[[row, pivot]];
            for column in 0..2 * size {
                let pivot_value = augmented[[pivot, column]];
                augmented[[row, column]] -= factor * pivot_value;
            }
        }
    }
    Array2::from_shape_fn((size, size), |(row, column)| {
        augmented[[row, size + column]]
    })
}
