//! Non-linear thermomechanical finite-element model of a conical disc spring.
//!
//! The meridian is discretised into `n_r` conical ring elements, each with an
//! `n_z`-point Gauss rule through the thickness. Kinematics follow the
//! Almen–László / DIN 2092 hypothesis (rigid meridian rotating about the
//! neutral radius `r₀`; the upper surface `ζ = +t/2` lies radially outside
//! the mid-surface) but with the *exact* rotation `φ₀ → φ`, so geometric
//! non-linearity is retained to all orders rather than truncated to the DIN
//! polynomial. Circumferential strain is `ε_θ = (r' − r)/r`, the constitutive
//! law is plane-stress `σ_θ = E(T) ε_θ / (1 − ν²)`, and thermal loading enters
//! through `E(T)` and the free thermal expansion `∫α(T)dT` of the geometry.
//!
//! Force control is solved by Newton–Raphson on the residual
//! `R(φ) = ∂U/∂φ − P ∂s/∂φ`; displacement control evaluates `P = ∂U/∂s`
//! directly. Both derivatives are analytic at the Gauss points.

use crate::belleville::DiscSpring;
use crate::material::{LinearCte, LinearModulus};
use crate::quadrature::gauss_legendre;

/// Thermomechanical disc model.
#[derive(Clone, Debug, PartialEq)]
pub struct ConicalDiscFe {
    /// Cold (reference) geometry.
    pub disc: DiscSpring,
    /// Modulus law `E(T)`.
    pub modulus: LinearModulus,
    /// Expansion law `α(T)`.
    pub cte: LinearCte,
    /// Reference temperature of the cold geometry [K].
    pub t_ref: f64,
    /// Meridional elements.
    pub n_r: usize,
    /// Through-thickness Gauss points.
    pub n_z: usize,
}

/// Per-element output of a solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElementState {
    /// Mid-surface radius of the element [m].
    pub radius: f64,
    /// Circumferential stress at the upper surface (`ζ = +t/2`) [Pa].
    pub sigma_theta_top: f64,
    /// Circumferential stress at the lower surface (`ζ = −t/2`) [Pa].
    pub sigma_theta_bottom: f64,
}

/// Converged equilibrium state.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscSolution {
    /// Axial deflection of the (thermally expanded) disc [m].
    pub deflection: f64,
    /// Axial force [N].
    pub force: f64,
    /// Tangent stiffness `dP/ds` [N/m].
    pub stiffness: f64,
    /// Newton iterations used (0 for displacement control).
    pub iterations: u32,
    /// Element stress states, inner to outer.
    pub elements: Vec<ElementState>,
}

/// Newton–Raphson failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolveError {
    /// Residual did not fall below tolerance within the iteration budget.
    NoConvergence,
    /// Requested force exceeds the flat-pressed load (no equilibrium on the
    /// rising branch).
    BeyondFlat,
}

struct HotGeometry {
    r0: f64,
    xi: Vec<f64>,
    w_xi: Vec<f64>,
    zeta: Vec<f64>,
    w_zeta: Vec<f64>,
    phi0: f64,
    span: f64,
    plate_modulus: f64,
    thickness: f64,
}

impl ConicalDiscFe {
    /// Model with default discretisation (64 meridional elements, 6 Gauss
    /// points through the thickness).
    pub fn new(disc: DiscSpring, modulus: LinearModulus, cte: LinearCte, t_ref: f64) -> Self {
        Self {
            disc,
            modulus,
            cte,
            t_ref,
            n_r: 64,
            n_z: 6,
        }
    }

    /// Geometry at temperature `t`, uniformly scaled by `1 + ∫α dT`.
    pub fn hot_disc(&self, t: f64) -> DiscSpring {
        let g = 1.0 + self.cte.strain(self.t_ref, t);
        DiscSpring {
            outer_diameter: self.disc.outer_diameter * g,
            inner_diameter: self.disc.inner_diameter * g,
            thickness: self.disc.thickness * g,
            cone_height: self.disc.cone_height * g,
        }
    }

    fn geometry(&self, t: f64) -> HotGeometry {
        let disc = self.hot_disc(t);
        let re = 0.5 * disc.outer_diameter;
        let ri = 0.5 * disc.inner_diameter;
        let r0 = disc.neutral_radius();
        let span = re - ri;
        let phi0 = (disc.cone_height / span).atan();
        let (gx, gw) = gauss_legendre(self.n_z);
        let (ex, ew) = gauss_legendre(4);
        let a = (ri - r0) / phi0.cos();
        let b = (re - r0) / phi0.cos();
        let h = (b - a) / self.n_r as f64;
        let mut xi = Vec::with_capacity(self.n_r * 4);
        let mut w_xi = Vec::with_capacity(self.n_r * 4);
        for e in 0..self.n_r {
            let lo = a + h * e as f64;
            for (x, w) in ex.iter().zip(&ew) {
                xi.push(lo + 0.5 * h * (1.0 + x));
                w_xi.push(0.5 * h * w);
            }
        }
        let half_t = 0.5 * disc.thickness;
        HotGeometry {
            r0,
            xi,
            w_xi,
            zeta: gx.iter().map(|x| half_t * x).collect(),
            w_zeta: gw.iter().map(|w| half_t * w).collect(),
            phi0,
            span,
            plate_modulus: self.modulus.at(t).plate_modulus(),
            thickness: disc.thickness,
        }
    }

    /// Strain energy derivatives `(∂U/∂φ, ∂²U/∂φ²)` at cone angle `phi`.
    fn energy_derivatives(g: &HotGeometry, phi: f64) -> (f64, f64) {
        let (c0, s0) = (g.phi0.cos(), g.phi0.sin());
        let (c, s) = (phi.cos(), phi.sin());
        let mut du = 0.0;
        let mut d2u = 0.0;
        for (xi, wx) in g.xi.iter().zip(&g.w_xi) {
            for (zeta, wz) in g.zeta.iter().zip(&g.w_zeta) {
                let r = g.r0 + xi * c0 + zeta * s0;
                let rp = g.r0 + xi * c + zeta * s;
                let eps = (rp - r) / r;
                let deps = (-xi * s + zeta * c) / r;
                let d2eps = (-xi * c - zeta * s) / r;
                let vol = 2.0 * std::f64::consts::PI * r * wx * wz;
                du += g.plate_modulus * eps * deps * vol;
                d2u += g.plate_modulus * (deps * deps + eps * d2eps) * vol;
            }
        }
        (du, d2u)
    }

    fn phi_of_deflection(g: &HotGeometry, s: f64) -> f64 {
        ((g.span * g.phi0.tan() - s) / g.span).atan()
    }

    fn deflection_of_phi(g: &HotGeometry, phi: f64) -> f64 {
        g.span * (g.phi0.tan() - phi.tan())
    }

    /// `ds/dφ = −span · sec²φ`.
    fn ds_dphi(g: &HotGeometry, phi: f64) -> f64 {
        -g.span / (phi.cos() * phi.cos())
    }

    fn solution(&self, g: &HotGeometry, phi: f64, iterations: u32) -> DiscSolution {
        let (du, d2u) = Self::energy_derivatives(g, phi);
        let dsdp = Self::ds_dphi(g, phi);
        let d2s = -2.0 * g.span * phi.tan() / (phi.cos() * phi.cos());
        let force = du / dsdp;
        let stiffness = (d2u - force * d2s) / (dsdp * dsdp);
        let (c0, s0) = (g.phi0.cos(), g.phi0.sin());
        let (c, s) = (phi.cos(), phi.sin());
        let per_elem = g.xi.len() / self.n_r;
        let half_t = 0.5 * g.thickness;
        let elements = (0..self.n_r)
            .map(|e| {
                let seg = &g.xi[e * per_elem..(e + 1) * per_elem];
                let xi_mid = 0.5 * (seg[0] + seg[per_elem - 1]);
                let stress = |zeta: f64| {
                    let r = g.r0 + xi_mid * c0 + zeta * s0;
                    let rp = g.r0 + xi_mid * c + zeta * s;
                    g.plate_modulus * (rp - r) / r
                };
                ElementState {
                    radius: g.r0 + xi_mid * c0,
                    sigma_theta_top: stress(half_t),
                    sigma_theta_bottom: stress(-half_t),
                }
            })
            .collect();
        DiscSolution {
            deflection: Self::deflection_of_phi(g, phi),
            force,
            stiffness,
            iterations,
            elements,
        }
    }

    /// Displacement-controlled solve at deflection `s` and temperature `t`.
    pub fn solve_deflection(&self, s: f64, t: f64) -> DiscSolution {
        let g = self.geometry(t);
        let phi = Self::phi_of_deflection(&g, s);
        self.solution(&g, phi, 0)
    }

    /// Force-controlled Newton–Raphson solve on the rising branch.
    pub fn solve_force(&self, force: f64, t: f64) -> Result<DiscSolution, SolveError> {
        let g = self.geometry(t);
        let flat = self.solution(&g, 0.0, 0);
        if force > flat.force * (1.0 + 1e-12) {
            return Err(SolveError::BeyondFlat);
        }
        let mut phi = Self::phi_of_deflection(&g, 0.5 * self.hot_disc(t).cone_height);
        let scale = flat.force.abs().max(1.0);
        for it in 1..=100 {
            let (du, d2u) = Self::energy_derivatives(&g, phi);
            let dsdp = Self::ds_dphi(&g, phi);
            let d2s = -2.0 * g.span * phi.tan() / (phi.cos() * phi.cos());
            let residual = du - force * dsdp;
            if residual.abs() / (scale * g.span) < 1e-13 {
                return Ok(self.solution(&g, phi, it));
            }
            let jac = d2u - force * d2s;
            let mut step = -residual / jac;
            let max_step = 0.25 * g.phi0;
            if step.abs() > max_step {
                step = max_step.copysign(step);
            }
            phi = (phi + step).clamp(0.0, g.phi0);
        }
        Err(SolveError::NoConvergence)
    }
}

/// Coffin–Manson strain–life relation `Δε_p = ε'_f N_f^{−c}` (cf.pdf Eq. 251)
/// solved for cycles to failure.
pub fn coffin_manson_cycles(plastic_strain_range: f64, eps_f: f64, c: f64) -> f64 {
    (eps_f / plastic_strain_range).powf(1.0 / c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belleville::DIN_2093_GROUP2;
    use crate::material::{T_COLD, T_HOT};

    fn model(disc: DiscSpring) -> ConicalDiscFe {
        ConicalDiscFe::new(
            disc,
            LinearModulus::DIN_2093_STEEL,
            LinearCte::INCONEL_X750,
            293.15,
        )
    }

    #[test]
    fn fe_curve_tracks_din_2092_group2_within_1_5_percent() {
        let e = LinearModulus::DIN_2093_STEEL.at(293.15);
        for entry in DIN_2093_GROUP2 {
            let disc = entry.disc();
            let fe = model(disc);
            for frac in [0.25, 0.5, 0.75, 1.0] {
                let s = frac * disc.cone_height;
                let sol = fe.solve_deflection(s, 293.15);
                let din = disc.force(s, e);
                let rel = (sol.force / din - 1.0).abs();
                assert!(
                    rel < 1.5e-2,
                    "{} s/h0={frac}: FE {:.1} vs DIN {:.1} ({:.3}%)",
                    entry.designation,
                    sol.force,
                    din,
                    rel * 100.0
                );
            }
        }
    }

    #[test]
    fn fe_stiffness_matches_finite_difference() {
        let disc = DIN_2093_GROUP2[5].disc();
        let fe = model(disc);
        let s = 0.4 * disc.cone_height;
        let h = 1e-8;
        let fd = (fe.solve_deflection(s + h, 293.15).force
            - fe.solve_deflection(s - h, 293.15).force)
            / (2.0 * h);
        let k = fe.solve_deflection(s, 293.15).stiffness;
        assert!((fd / k - 1.0).abs() < 1e-5, "{fd} vs {k}");
    }

    #[test]
    fn force_control_inverts_displacement_control() {
        let disc = DIN_2093_GROUP2[15].disc();
        let fe = model(disc);
        let s = 0.6 * disc.cone_height;
        let target = fe.solve_deflection(s, 293.15).force;
        let sol = fe.solve_force(target, 293.15).unwrap();
        assert!((sol.deflection - s).abs() < 1e-12 * disc.cone_height);
        assert!(sol.iterations <= 12, "{}", sol.iterations);
        assert!(matches!(
            fe.solve_force(10.0 * target, 293.15),
            Err(SolveError::BeyondFlat)
        ));
    }

    #[test]
    fn thermal_softening_and_growth_across_operating_band() {
        let disc = DiscSpring {
            outer_diameter: 0.250,
            inner_diameter: 0.125,
            thickness: 12.0e-3,
            cone_height: 17.4e-3,
        };
        let fe = ConicalDiscFe::new(
            disc,
            LinearModulus::INCONEL_X750,
            LinearCte::INCONEL_X750,
            T_COLD,
        );
        let s = 4.5625e-3;
        let cold = fe.solve_deflection(s, T_COLD);
        let hot = fe.solve_deflection(s, T_HOT);
        assert!(hot.force < cold.force);
        let e_ratio = LinearModulus::INCONEL_X750.at(T_HOT).youngs
            / LinearModulus::INCONEL_X750.at(T_COLD).youngs;
        let ratio = hot.force / cold.force;
        assert!((ratio - e_ratio).abs() < 0.02, "{ratio} vs {e_ratio}");
        let grown = fe.hot_disc(T_HOT);
        let g = 1.0 + 14.2e-6 * 325.0;
        assert!((grown.outer_diameter / disc.outer_diameter - g).abs() < 1e-15);
        assert!(cold.elements.len() == 64);
        assert!(cold.elements[0].sigma_theta_top < 0.0);
    }

    #[test]
    fn coffin_manson_roundtrip() {
        let n = coffin_manson_cycles(0.004, 0.35, 0.6);
        let strain = 0.35 * n.powf(-0.6);
        assert!((strain - 0.004).abs() < 1e-15);
    }
}
