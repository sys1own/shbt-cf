//! Spatio-temporal full-vector Maxwell blocks.
//!
//! The Update-8 default has 625 spatial harmonics and 15 temporal sidebands.
//! The matrix is retained as 16 faer blocks: this is the explicit algebraic
//! system without forcing an unusable 37,500² dense allocation.

use crate::cmat::{diag_re, CMat};
use faer::linalg::solvers::DenseSolveCore;
use faer::prelude::*;

/// Update-8 truncation: 25 x 25 spatial orders and 15 temporal sidebands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FloquetTruncation {
    /// Number of spatial orders along x.
    pub spatial_x: usize,
    /// Number of spatial orders along y.
    pub spatial_y: usize,
    /// Number of retained temporal sidebands.
    pub temporal: usize,
}

impl FloquetTruncation {
    /// The Update-8 production truncation.
    pub const UPDATE_8: Self = Self {
        spatial_x: 25,
        spatial_y: 25,
        temporal: 15,
    };

    /// Number of composite harmonics.
    pub fn harmonics(self) -> usize {
        self.spatial_x * self.spatial_y * self.temporal
    }

    /// Number of transverse field unknowns in the first-order system.
    pub fn system_dimension(self) -> usize {
        4 * self.harmonics()
    }
}

/// Laurent-Li convolution blocks for anisotropic permittivity/permeability.
#[derive(Clone, Debug)]
pub struct TensorConvolution {
    /// Permittivity blocks in row-major tensor order.
    pub epsilon: [[CMat; 3]; 3],
    /// Permeability blocks in row-major tensor order.
    pub mu: [[CMat; 3]; 3],
}

/// Explicit 4 x 4 block representation of the transverse Maxwell matrix.
#[derive(Clone, Debug)]
pub struct MaxwellSystemMatrix {
    /// Four-by-four H x H blocks in `(e_x,e_y,h_x,h_y)` order.
    pub blocks: [[CMat; 4]; 4],
    /// Number of composite harmonics H.
    pub harmonic_dimension: usize,
}

impl MaxwellSystemMatrix {
    /// Dense materialisation for reduced verification systems only.
    pub fn to_dense(&self) -> CMat {
        let h = self.harmonic_dimension;
        Mat::from_fn(4 * h, 4 * h, |i, j| {
            self.blocks[i / h][j / h][(i % h, j % h)]
        })
    }
}

fn neg(a: &CMat) -> CMat {
    -a
}
/// Assemble the exact transverse block system after eliminating `E_z,H_z`.
pub fn assemble_maxwell_system(
    tensors: &TensorConvolution,
    kx: &[f64],
    ky: &[f64],
    frequency_ratio: &[f64],
) -> MaxwellSystemMatrix {
    let h = kx.len();
    assert_eq!(ky.len(), h);
    assert_eq!(frequency_ratio.len(), h);
    for tensor in tensors.epsilon.iter().chain(tensors.mu.iter()) {
        for block in tensor {
            assert_eq!((block.nrows(), block.ncols()), (h, h));
        }
    }
    let kx = diag_re(kx);
    let ky = diag_re(ky);
    let w = diag_re(frequency_ratio);
    let ei = tensors.epsilon[2][2].partial_piv_lu().inverse();
    let mi = tensors.mu[2][2].partial_piv_lu().inverse();
    let e = &tensors.epsilon;
    let m = &tensors.mu;
    let b = [
        [
            neg(&(&kx * &ei * &e[2][0])) + &m[1][2] * &mi * &ky,
            neg(&(&kx * &ei * &e[2][1])) - &m[1][2] * &mi * &kx,
            &kx * &ei * &ky + &w * &m[1][0] - &m[1][2] * &mi * &m[2][0],
            -(&kx * &ei * &kx) + &w * &m[1][1] - &m[1][2] * &mi * &m[2][1],
        ],
        [
            neg(&(&ky * &ei * &e[2][0])) + &m[0][2] * &mi * &ky,
            neg(&(&ky * &ei * &e[2][1])) - &m[0][2] * &mi * &kx,
            &ky * &ei * &ky - &w * &m[0][0] - &m[0][2] * &mi * &m[2][0],
            -(&ky * &ei * &kx) - &w * &m[0][1] - &m[0][2] * &mi * &m[2][1],
        ],
        [
            -(&ky * &mi * &ky) + &w * &e[1][0] - &e[1][2] * &ei * &e[2][0],
            &ky * &mi * &kx + &w * &e[1][1] - &e[1][2] * &ei * &e[2][1],
            -(&ky * &mi * &m[2][0]) + &e[1][2] * &ei * &ky,
            -(&ky * &mi * &m[2][1]) - &e[1][2] * &ei * &kx,
        ],
        [
            &kx * &mi * &ky - &w * &e[0][0] - &e[0][2] * &ei * &e[2][0],
            -(&kx * &mi * &kx) - &w * &e[0][1] - &e[0][2] * &ei * &e[2][1],
            &kx * &mi * &m[2][0] + &e[0][2] * &ei * &ky,
            &kx * &mi * &m[2][1] - &e[0][2] * &ei * &kx,
        ],
    ];
    MaxwellSystemMatrix {
        blocks: b,
        harmonic_dimension: h,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cmat::{c64, ZERO};

    fn tensors(n: usize) -> TensorConvolution {
        let make = |value| {
            Mat::from_fn(
                n,
                n,
                |i, j| if i == j { c64::new(value, 0.0) } else { ZERO },
            )
        };
        TensorConvolution {
            epsilon: std::array::from_fn(|i| {
                std::array::from_fn(|j| make(if i == j { 2.0 } else { 0.0 }))
            }),
            mu: std::array::from_fn(|i| {
                std::array::from_fn(|j| make(if i == j { 1.0 } else { 0.0 }))
            }),
        }
    }

    #[test]
    fn update8_dimension_is_explicit_without_dense_allocation() {
        assert_eq!(FloquetTruncation::UPDATE_8.harmonics(), 9375);
        assert_eq!(FloquetTruncation::UPDATE_8.system_dimension(), 37500);
    }

    #[test]
    fn isotropic_reduced_system_has_expected_shape() {
        let system = assemble_maxwell_system(&tensors(2), &[0.0, 0.1], &[0.0, 0.2], &[1.0, 1.1]);
        assert_eq!(system.to_dense().nrows(), 8);
        assert!(system.to_dense()[(0, 0)].re.is_finite());
    }
}
