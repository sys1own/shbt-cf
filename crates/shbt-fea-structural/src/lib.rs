//! Non-linear 3D thermomechanical finite-element models: DIN 2092/2093
//! Belleville washer stacks, extended Stoney thin-film stress and
//! Coffin-Manson low-cycle fatigue (simulator_spec.pdf §3, §4).
//!
//! Depends exclusively on `shbt-core-math`. Solvers are implemented in Stage 3.

#![forbid(unsafe_code)]

use shbt_core_math::precision::PrecisionMode;

/// Canonical crate identifier used by the workbench for topology reporting.
pub const CRATE_NAME: &str = "shbt-fea-structural";

/// Real-time dynamic FEA solves run in strict binary64 (spec §1 precision table).
pub const DYNAMIC_SOLVE_PRECISION: PrecisionMode = PrecisionMode::StrictF64;

/// Belleville stack layout: `parallel` washers share force in each group,
/// `series` groups share deflection (DIN 2092 compound stacks).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StackLayout {
    /// Washers nested in parallel per group.
    pub parallel: u32,
    /// Groups arranged in series.
    pub series: u32,
}

impl StackLayout {
    /// Total force of the stack given the force of a single washer at the
    /// per-group deflection.
    pub fn stack_force(self, single_washer_force: f64) -> f64 {
        single_washer_force * self.parallel as f64
    }

    /// Total deflection of the stack given a single group's deflection.
    pub fn stack_deflection(self, group_deflection: f64) -> f64 {
        group_deflection * self.series as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compound_stack_scaling() {
        let layout = StackLayout {
            parallel: 2,
            series: 3,
        };
        assert_eq!(layout.stack_force(100.0), 200.0);
        assert_eq!(layout.stack_deflection(0.5), 1.5);
    }
}
