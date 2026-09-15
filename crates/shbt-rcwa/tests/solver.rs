#![allow(missing_docs)]

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use shbt_rcwa::{BoundaryDiffractionSolver, LayerSMatrix, RcwaSystemSolver};

fn diagonal(value: f64, size: usize) -> Array2<Complex64> {
    Array2::from_shape_fn((size, size), |(i, j)| {
        if i == j {
            Complex64::new(value, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    })
}

#[test]
fn identity_cascade_is_identity() {
    let solver = RcwaSystemSolver::with_harmonics(2, 2, 500e-9);
    let identity = solver.identity_s_matrix();
    let result = solver.star_product_cascade(&identity, &identity);
    assert_eq!(result, identity);
}

#[test]
fn cascade_matches_scalar_redheffer_blocks() {
    let solver = RcwaSystemSolver::with_harmonics(1, 1, 500e-9);
    let a = LayerSMatrix {
        s11: diagonal(0.8, 2),
        s12: diagonal(0.1, 2),
        s21: diagonal(0.2, 2),
        s22: diagonal(0.7, 2),
    };
    let b = LayerSMatrix {
        s11: diagonal(0.6, 2),
        s12: diagonal(0.05, 2),
        s21: diagonal(0.3, 2),
        s22: diagonal(0.9, 2),
    };
    let result = solver.star_product_cascade(&a, &b);
    let denominator = 1.0 - 0.1 * 0.3;
    assert!((result.s11[[0, 0]].re - 0.6 * 0.8 / denominator).abs() < 1e-12);
    assert!((result.s22[[0, 0]].re - 0.7 * 0.9 / denominator).abs() < 1e-12);
}

#[test]
fn flat_surface_normals_point_along_z() {
    let solver = RcwaSystemSolver::with_harmonics(2, 2, 500e-9);
    let normals = solver.compute_normal_vector_field(&Array2::zeros((3, 4)));
    assert!(normals
        .iter()
        .enumerate()
        .all(
            |(index, value)| index % 3 == 2 && (*value - 1.0).abs() < 1e-12
                || index % 3 != 2 && value.abs() < 1e-12
        ));
}

#[test]
fn li_factorization_produces_transverse_convolution_block() {
    let solver = RcwaSystemSolver::with_harmonics(2, 2, 500e-9);
    let mut epsilon = Array3::zeros((2, 2, 9));
    for mut sample in epsilon.outer_iter_mut() {
        for mut row in sample.outer_iter_mut() {
            row[0] = Complex64::new(2.0, 0.0);
            row[4] = Complex64::new(2.0, 0.0);
            row[8] = Complex64::new(1.0, 0.0);
        }
    }
    let normals = solver.compute_normal_vector_field(&Array2::zeros((2, 2)));
    let factorized = solver.apply_li_factorization(&epsilon, &normals);
    assert_eq!(factorized.dim(), (8, 8));
    assert!(factorized[[0, 0]].re > 0.0);
}
