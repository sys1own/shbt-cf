//! Inverse Floquet-Adler-Wiser solver for non-local dielectric response
//! tensors of periodic boundary media (simulator_spec.pdf §2, §3).
//!
//! Depends exclusively on `shbt-core-math`. Fourier-space matrix assembly and
//! the arbitrary-precision inverse are implemented in Stage 3 of the blueprint.

#![forbid(unsafe_code)]

use shbt_core_math::precision::PrecisionMode;

/// Canonical crate identifier used by the workbench for topology reporting.
pub const CRATE_NAME: &str = "shbt-dielectric-floquet";

/// Residual convergence target for the inverse Floquet iteration.
pub const RESIDUAL_TOLERANCE: f64 = 1e-12;

/// Selects the arithmetic tier for inverting a Floquet matrix with the given
/// estimated condition number.
pub fn inversion_precision(condition_number: f64) -> PrecisionMode {
    PrecisionMode::arbitrary_for_condition_number(condition_number)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn well_conditioned_uses_minimum_width() {
        assert_eq!(inversion_precision(10.0).significant_bits(), 128);
    }
}
