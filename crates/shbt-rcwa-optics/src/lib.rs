//! 2D/3D Rigorous Coupled-Wave Analysis with scattering-matrix (S-matrix)
//! recursion for thick grating stacks (simulator_spec.pdf §3, §4).
//!
//! Depends exclusively on `shbt-core-math`. Eigenvalue formulation and the
//! Redheffer star-product S-matrix composition are implemented in Stage 3.

#![forbid(unsafe_code)]

use shbt_core_math::fixed::Q64x64;

/// Canonical crate identifier used by the workbench for topology reporting.
pub const CRATE_NAME: &str = "shbt-rcwa-optics";

/// Drift-free optical phase accumulator in units of turns (1.0 == 2π).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhaseAccumulator {
    turns: Q64x64,
}

impl PhaseAccumulator {
    /// Creates an accumulator at zero phase.
    pub const fn new() -> Self {
        Self {
            turns: Q64x64::ZERO,
        }
    }

    /// Advances the phase by `delta_turns`, wrapping the integer turn count so
    /// the accumulator never overflows.
    pub fn advance(&mut self, delta_turns: Q64x64) {
        let sum = self.turns.checked_add(delta_turns).unwrap_or(Q64x64::ZERO);
        self.turns = Q64x64::from_bits(sum.to_bits() & (u64::MAX as i128));
    }

    /// Current phase in `[0, 1)` turns.
    pub fn turns(&self) -> Q64x64 {
        self.turns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_to_unit_interval() {
        let mut acc = PhaseAccumulator::new();
        let step = Q64x64::from_f64(0.25).unwrap();
        for _ in 0..5 {
            acc.advance(step);
        }
        assert_eq!(acc.turns(), Q64x64::from_f64(0.25).unwrap());
    }
}
