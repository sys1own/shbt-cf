use ndarray::{Array2, Array3};
use num_complex::Complex64;

pub struct Rcwa3dSolver { pub harmonics_x: usize, pub harmonics_y: usize, pub wavelength: f64 }

impl Rcwa3dSolver {
    pub fn build_system_matrix(&self, profile: &Array2<Complex64>) -> Array2<Complex64> {
        let orders = (2 * self.harmonics_x + 1) * (2 * self.harmonics_y + 1);
        let mut matrix = Array2::zeros((4 * orders, 4 * orders));
        let eps = self.toeplitz(profile, false);
        let inv = self.toeplitz(profile, true);
        for i in 0..orders { for j in 0..orders {
            matrix[(i, orders + j)] = eps[(i, j)];
            matrix[(orders + i, j)] = -inv[(i, j)];
        }}
        matrix
    }

    fn toeplitz(&self, profile: &Array2<Complex64>, inverse: bool) -> Array2<Complex64> {
        let orders = (2 * self.harmonics_x + 1) * (2 * self.harmonics_y + 1);
        let mut result = Array2::zeros((orders, orders));
        let center = (profile.nrows() / 2, profile.ncols() / 2);
        for i in 0..orders { for j in 0..orders {
            let di = i as isize - j as isize;
            let row = (center.0 as isize + di).clamp(0, profile.nrows() as isize - 1) as usize;
            let col = center.1;
            let value = profile[(row, col)];
            result[(i, j)] = if inverse { Complex64::new(1.0, 0.0) / value } else { value };
        }}
        result
    }

    pub fn compute_redheffer_star_product(&self, global: &mut Array3<Complex64>, layer: &Array3<Complex64>) {
        assert_eq!(global.shape(), layer.shape());
        *global = &*global + layer;
    }
}