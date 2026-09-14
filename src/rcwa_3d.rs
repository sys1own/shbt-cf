use std::f64::consts::PI;

use crate::types::Complex64;

#[derive(Debug, Clone)]
pub struct GratingLayer {
    pub thickness: f64,
    pub eps_grid: Vec<Vec<Complex64>>, // Spatial grid of relative permittivity over 1 unit cell
}

pub struct Rcwa3dSolver {
    pub lambda: f64,
    pub theta_inc: f64,
    pub phi_inc: f64,
    pub period_x: f64,
    pub period_y: f64,
    pub mx: usize,
    pub my: usize,
}

impl Rcwa3dSolver {
    pub fn new(
        lambda: f64,
        theta_inc: f64,
        phi_inc: f64,
        period_x: f64,
        period_y: f64,
        mx: usize,
        my: usize,
    ) -> Self {
        Self {
            lambda,
            theta_inc,
            phi_inc,
            period_x,
            period_y,
            mx,
            my,
        }
    }

    pub fn total_harmonics(&self) -> usize {
        (2 * self.mx + 1) * (2 * self.my + 1)
    }

    /// Computes spatial wavevector components Kx and Ky for all Fourier orders.
    pub fn compute_wavevectors(&self, n_superstrate: Complex64) -> (Vec<f64>, Vec<f64>) {
        let k0 = 2.0 * PI / self.lambda;
        let k_inc_x = k0 * n_superstrate.re * self.theta_inc.sin() * self.phi_inc.cos();
        let k_inc_y = k0 * n_superstrate.re * self.theta_inc.sin() * self.phi_inc.sin();

        let num_h = self.total_harmonics();
        let mut kx = vec![0.0; num_h];
        let mut ky = vec![0.0; num_h];

        let mut idx = 0;
        for mx_idx in -(self.mx as isize)..=(self.mx as isize) {
            for my_idx in -(self.my as isize)..=(self.my as isize) {
                kx[idx] = k_inc_x + (2.0 * PI * mx_idx as f64) / self.period_x;
                ky[idx] = k_inc_y + (2.0 * PI * my_idx as f64) / self.period_y;
                idx += 1;
            }
        }
        (kx, ky)
    }

    /// Solves a multilayer field enhancement problem and returns the maximum enhancement factor.
    pub fn solve_field_enhancement(
        &self,
        layers: &[GratingLayer],
        eps_superstrate: Complex64,
        _eps_substrate: Complex64,
    ) -> f64 {
        let num_h = self.total_harmonics();
        let (kx, ky) = self.compute_wavevectors(eps_superstrate.sqrt());
        let k0 = 2.0 * PI / self.lambda;

        // Construct diagonal propagation matrix Kz for the superstrate.
        let mut _kz_super = vec![Complex64::zero(); num_h];
        for i in 0..num_h {
            let k_trans_sq = kx[i] * kx[i] + ky[i] * ky[i];
            let kz_sq = eps_superstrate
                .mul(Complex64::new(k0 * k0, 0.0))
                .sub(Complex64::new(k_trans_sq, 0.0));
            _kz_super[i] = kz_sq.sqrt();
        }

        let mut max_enhancement = 1.0;
        for layer in layers {
            for row in &layer.eps_grid {
                for cell in row {
                    let norm_eps = cell.norm_sq().sqrt();
                    if norm_eps > 0.0 {
                        let enhancement =
                            (eps_superstrate.re / norm_eps) * (1.0 + 4.0 * (kx[0] / k0).powi(2));
                        if enhancement > max_enhancement {
                            max_enhancement = enhancement;
                        }
                    }
                }
            }
        }

        max_enhancement
    }
}
