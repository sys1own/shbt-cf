//! Variational conical-shell preload with unilateral contact.

use crate::material::Elastic;

/// Discretized shell parameters for the Zheng-style energy functional.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VariationalDisc {
    /// Inner and outer radii [m].
    pub inner_radius: f64,
    /// Outer radius [m].
    pub outer_radius: f64,
    /// Thickness and free cone height [m].
    pub thickness: f64,
    /// Free cone height [m].
    pub cone_height: f64,
    /// Initial cone angle [rad].
    pub angle: f64,
    /// Isotropic elastic constants.
    pub elastic: Elastic,
    /// Contact friction coefficient.
    pub friction: f64,
}

impl VariationalDisc {
    fn energy(&self, deflection: f64) -> f64 {
        let span = self.outer_radius - self.inner_radius;
        let membrane = self.elastic.plate_modulus()
            * self.thickness
            * span
            * (deflection / span - self.angle.sin()).powi(2)
            * std::f64::consts::PI;
        let bending = self.elastic.plate_modulus() * self.thickness.powi(3) * deflection.powi(2)
            / (12.0 * span)
            * std::f64::consts::PI;
        membrane + bending
    }

    /// Total potential energy including external work and friction dissipation.
    pub fn potential(&self, deflection: f64, force: f64) -> f64 {
        self.energy(deflection) - force * deflection
            + self.friction.abs() * force.abs() * deflection.abs()
    }

    /// Equilibrium force from `δΠ = 0`, with a Signorini reaction at flat contact.
    pub fn equilibrium_force(&self, deflection: f64) -> f64 {
        let span = self.outer_radius - self.inner_radius;
        let membrane_slope = self.elastic.plate_modulus()
            * self.thickness
            * 2.0
            * std::f64::consts::PI
            * (deflection / span - self.angle.sin());
        let bending_slope = self.elastic.plate_modulus()
            * self.thickness.powi(3)
            * deflection
            * std::f64::consts::PI
            / (6.0 * span);
        (membrane_slope + bending_slope).max(0.0)
    }

    /// Solves the scalar stationarity condition by safeguarded Newton iteration.
    pub fn solve(&self, target_force: f64) -> f64 {
        let mut deflection = 0.5 * self.cone_height;
        for _ in 0..64 {
            let residual = self.equilibrium_force(deflection) - target_force;
            if residual.abs() < target_force.abs().max(1.0) * 1e-12 {
                break;
            }
            let h = self.thickness * 1e-5;
            let slope = (self.equilibrium_force(deflection + h)
                - self.equilibrium_force(deflection - h))
                / (2.0 * h);
            deflection =
                (deflection - residual / slope.max(f64::MIN_POSITIVE)).clamp(0.0, self.cone_height);
        }
        deflection
    }
}
