//! 2D/3D Rigorous Coupled-Wave Analysis with scattering-matrix (S-matrix)
//! recursion for periodic sub-wavelength grating stacks
//! (simulator_spec.pdf §3–§4; cf.pdf §"Pump Light Absorption" Option B).
//!
//! * [`grating`] — harmonic truncation, lamellar / crossed unit cells,
//!   Li-factorised convolution matrices.
//! * [`solver`] — eigenmodes, Redheffer star product, order efficiencies,
//!   interface and internal field reconstruction.
//! * [`option_b`] — reactor grating: Λ = 960.80 nm relief, dual-pump
//!   785/802.5 nm, specular reflectivity and `|E/E₀|²` enhancement.
//! * [`material`] — Table VIII optical constants.
//!
//! Depends exclusively on `shbt-core-math`.

#![forbid(unsafe_code)]

pub mod cmat;
pub mod grating;
pub mod material;
pub mod option_b;
pub mod solver;
pub mod spacetime;

pub use grating::{Profile, Segment, Truncation};
pub use solver::{EField, Excitation, Lattice, Layer, SMatrix, Scattering, Side, Stack};

use shbt_core_math::fixed::Q64x64;

/// Canonical crate identifier used by the workbench for topology reporting.
pub const CRATE_NAME: &str = "shbt-rcwa-optics";

/// Drift-free optical phase accumulator in units of turns (1.0 == 2π).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhaseAccumulator {
    turns: Q64x64,
}

impl PhaseAccumulator {
    /// Zeroed accumulator.
    pub const fn new() -> Self {
        Self {
            turns: Q64x64::ZERO,
        }
    }

    /// Advances the phase by `delta_turns` modulo one turn.
    pub fn advance(&mut self, delta_turns: Q64x64) {
        let sum = self.turns.checked_add(delta_turns).unwrap_or(Q64x64::ZERO);
        self.turns = Q64x64::from_bits(sum.to_bits() & (u64::MAX as i128));
    }

    /// Accumulated phase in turns.
    pub fn turns(&self) -> Q64x64 {
        self.turns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_wraps_in_turns() {
        let mut p = PhaseAccumulator::new();
        p.advance(Q64x64::from_f64(0.75).unwrap());
        p.advance(Q64x64::from_f64(0.5).unwrap());
        assert_eq!(p.turns(), Q64x64::from_f64(0.25).unwrap());
    }
}
