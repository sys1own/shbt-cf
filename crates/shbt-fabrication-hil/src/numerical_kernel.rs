use rug::Float;

/// High-precision checks used by the Python orchestration layer.
#[derive(Debug, Clone, Copy)]
pub struct ShbtNumericalKernel {
    /// Precision used for arbitrary-precision calculations.
    pub precision_bits: u32,
}

impl Default for ShbtNumericalKernel {
    fn default() -> Self {
        Self {
            precision_bits: 512,
        }
    }
}

impl ShbtNumericalKernel {
    /// Verify the four retained Hankel singular values of the 15-to-4 model.
    pub fn verify_state_reduction(&self, hsvs: &[f64]) -> Result<bool, String> {
        if hsvs.len() < 4 {
            return Err("Insufficient Hankel singular values.".to_owned());
        }
        let target = [12.45, 5.13, 1.85, 0.94];
        for (index, (expected, actual)) in target.iter().zip(&hsvs[..4]).enumerate() {
            if !actual.is_finite() || (actual - expected).abs() > 1e-12 {
                return Err(format!(
                    "AnomalyClosureError: HSV state {index} detuned. Rigidity broken."
                ));
            }
        }
        Ok(true)
    }

    /// Compute the effective dielectric inverse and binary64 drift.
    pub fn compute_floquet_inversion(
        &self,
        u_eff: f64,
        v_driven: f64,
    ) -> Result<(String, f64, String), String> {
        if !u_eff.is_finite() || !v_driven.is_finite() || u_eff.abs() < 1e-6 {
            return Err("Singularity: Static potential is zero or non-finite.".to_owned());
        }
        let ratio_f64 = v_driven / u_eff;
        let inv_f64 = 1.0 / (1.0 - ratio_f64 * ratio_f64);
        let u = Float::with_val(self.precision_bits, u_eff);
        let v = Float::with_val(self.precision_bits, v_driven);
        let ratio = Float::with_val(self.precision_bits, &v / &u);
        let ratio_sq = Float::with_val(self.precision_bits, &ratio * &ratio);
        let epsilon = Float::with_val(self.precision_bits, 1) - ratio_sq;
        if epsilon.is_zero() {
            return Err("Holographic singularity: Dielectric constant is zero.".to_owned());
        }
        let inverse = Float::with_val(self.precision_bits, 1) / &epsilon;
        let drift = (Float::with_val(self.precision_bits, inv_f64) - &inverse).abs();
        Ok((
            format!("{inverse:.150e}"),
            inv_f64,
            format!("{drift:.150e}"),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::ShbtNumericalKernel;

    #[test]
    fn verifies_reference_reduction_and_rejects_detuning() {
        let kernel = ShbtNumericalKernel::default();
        assert_eq!(
            kernel.verify_state_reduction(&[12.45, 5.13, 1.85, 0.94]),
            Ok(true)
        );
        assert!(kernel
            .verify_state_reduction(&[12.45000000001, 5.13, 1.85, 0.94])
            .is_err());
    }

    #[test]
    fn computes_reference_floquet_inversion() {
        let (high_precision, binary64, _) = ShbtNumericalKernel::default()
            .compute_floquet_inversion(350.0, -298.57)
            .expect("reference values are non-singular");
        assert!((binary64 - 3.672507641671456).abs() < 1e-12);
        assert!(high_precision.starts_with("3.672507641671"));
    }
}
