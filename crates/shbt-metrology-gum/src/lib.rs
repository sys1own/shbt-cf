//! ISO/IEC Guide 98-3 (GUM, Supplement 1) metrology analytics: covariance
//! propagation, dual-channel calorimetric uncertainty and parallel Monte Carlo
//! sampling (simulator_spec.pdf §2, §4).
//!
//! * [`distributions`] — marginal distributions and covariance-correlated
//!   Gaussian blocks (Cholesky-ingested `Σ`).
//! * [`engine`] — deterministic parallel Monte Carlo over `N ≥ 10⁶`
//!   evaluations with coverage intervals and per-input sensitivities;
//!   [`engine::models`] provides the net-power and Coffin–Manson measurands.
//! * [`random`] — counter-seeded Xoshiro256** for bit-reproducible runs.
//!
//! Depends exclusively on `shbt-core-math`.

#![forbid(unsafe_code)]

pub mod distributions;
pub mod engine;
pub mod random;

pub use distributions::{Input, Marginal, Sampler};
pub use engine::{propagate, McReport, Sensitivity};

use shbt_core_math::precision::PrecisionMode;

/// Canonical crate identifier used by the workbench for topology reporting.
pub const CRATE_NAME: &str = "shbt-metrology-gum";

/// Monte Carlo evaluation loops run in strict binary64.
pub const SAMPLING_PRECISION: PrecisionMode = PrecisionMode::StrictF64;

/// Combined standard uncertainty of a sum of independent inputs
/// (GUM eq. 10 with unit sensitivity coefficients).
pub fn combined_standard_uncertainty(standard_uncertainties: &[f64]) -> f64 {
    standard_uncertainties
        .iter()
        .map(|u| u * u)
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_sum_square() {
        assert_eq!(combined_standard_uncertainty(&[3.0, 4.0]), 5.0);
        assert_eq!(combined_standard_uncertainty(&[]), 0.0);
    }
}
