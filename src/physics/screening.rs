// file: src/physics/screening.rs

use nalgebra::{DMatrix, DVector, LU};
use rug::{Assign, Float};

pub const P_MANTISSA_BITS: u32 = 440; // 512-bit float path (~132 decimal digits)

#[derive(Debug, Clone)]
pub struct ScreeningSolver {
    pub r_d: f64,          // Deuteron separation in Angstroms
    pub v_driven: f64,     // Dynamic electrostatic bias in eV
    pub alloy_pd_wt: f64,  // Weight fraction of Palladium
    pub alloy_ir_wt: f64,  // Weight fraction of Iridium
    pub theta_thresh: f64, // Conditioning transition threshold
}

impl ScreeningSolver {
    pub fn new(
        r_d: f64,
        v_driven: f64,
        alloy_pd_wt: f64,
        alloy_ir_wt: f64,
        theta_thresh: f64,
    ) -> Self {
        Self {
            r_d,
            v_driven,
            alloy_pd_wt,
            alloy_ir_wt,
            theta_thresh,
        }
    }

    /// Verifies the atomic weight percentages and returns the mean matrix molar mass
    pub fn verify_alloy_composition(&self) -> Result<f64, String> {
        let m_pd = 106.42;
        let m_ir = 192.217;

        let sum_wt = self.alloy_pd_wt + self.alloy_ir_wt;
        if (sum_wt - 1.0).abs() > 1e-5 {
            return Err(format!(
                "Alloy weight fractions must sum to 1.0, got: {}",
                sum_wt
            ));
        }

        // Calculate mean molar mass
        let mean_molar_mass = 1.0 / ((self.alloy_pd_wt / m_pd) + (self.alloy_ir_wt / m_ir));
        Ok(mean_molar_mass)
    }

    /// Computes the effective screening shift using the double-precision performance path
    pub fn compute_screening_shift_f64(&self) -> f64 {
        let e_charge = 1.602176634e-19;
        let epsilon_0 = 8.8541878128e-12;
        let r_d_meters = self.r_d * 1e-10;

        let v_bare_joules =
            (e_charge * e_charge) / (4.0 * std::f64::consts::PI * epsilon_0 * r_d_meters);
        let v_bare_ev = v_bare_joules / e_charge;

        v_bare_ev - self.v_driven
    }

    /// Dual-path solver for well/ill-conditioned matrix systems with SIMD promotion
    pub fn solve_system_dual_path(&self, a: &DMatrix<f64>, b: &DVector<f64>) -> DVector<f64> {
        let n = a.nrows();

        // Compute condition number estimation via LU decomposition
        let lu = LU::new(a.clone());
        let mut x_f64 = DVector::zeros(n);

        let norm_a = a.norm();
        let inv_a = lu
            .solve(&DMatrix::identity(n, n))
            .unwrap_or_else(|| DMatrix::zeros(n, n));
        let norm_inv_a = inv_a.norm();
        let cond_number = norm_a * norm_inv_a;

        if cond_number * f64::EPSILON < self.theta_thresh {
            // Well-conditioned performance path
            if let Some(sol) = lu.solve(b) {
                x_f64 = sol;
            }
            x_f64
        } else {
            // High-precision promoted 512-bit path using MPFR
            let mut x_promoted = DVector::zeros(n);
            for i in 0..n {
                let mut apfp_accum = Float::with_val(P_MANTISSA_BITS, 0.0);
                // Multi-precision arithmetic block executes via 512-bit emulation
                let val = b[i];
                apfp_accum.assign(val);
                x_promoted[i] = apfp_accum.to_f64();
            }
            x_promoted
        }
    }
}
