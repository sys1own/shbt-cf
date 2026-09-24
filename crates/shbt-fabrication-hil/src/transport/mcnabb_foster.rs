#[derive(Clone, Debug)]
/// One-dimensional finite-difference mesh for deuterium transport.
pub struct TransportGrid {
    /// Uniform node spacing in metres.
    pub dx: f64,
    /// Number of concentration nodes.
    pub size: usize,
    /// Node indices where material interface conditions are applied.
    pub interface_boundaries: Vec<usize>,
}

#[derive(Clone, Debug)]
/// Material coefficients used by the transport discretisation.
pub struct MaterialProfile {
    /// Lattice diffusion coefficient in square metres per second.
    pub diffusion_coeff: f64,
    /// Lattice solubility in concentration units per site fraction.
    pub solubility: f64,
    /// Partial molar volume of deuterium in cubic metres per mole.
    pub partial_molar_volume: f64,
}

#[derive(Clone, Debug)]
/// Reversible McNabb-Foster trap population at one grid node.
pub struct TrapState {
    /// Density of trapping sites.
    pub density: f64,
    /// Capture rate from the lattice into the trap.
    pub capture_rate: f64,
    /// Release rate from the trap back to the lattice.
    pub release_rate: f64,
}

/// Explicit finite-difference solver for stress-assisted trapping transport.
pub struct McNabbFosterSolver {
    /// Spatial grid and interface locations.
    pub grid: TransportGrid,
    /// Material profile at each grid node.
    pub profiles: Vec<MaterialProfile>,
    /// Trap populations indexed by grid node and trap type.
    pub trap_populations: Vec<Vec<TrapState>>,
    /// Mobile lattice concentration at each grid node.
    pub c_l: Vec<f64>,
    /// Occupancy fraction for each trap population.
    pub theta: Vec<Vec<f64>>,
    /// Hydrostatic stress at each grid node in pascals.
    pub hydrostatic_stress: Vec<f64>,
    /// Absolute temperature in kelvin.
    pub temperature: f64,
}

impl McNabbFosterSolver {
    /// Advance the coupled lattice/trap state by `dt` seconds.
    pub fn step(&mut self, dt: f64) {
        assert!(dt > 0.0 && self.c_l.len() == self.grid.size);
        let n = self.grid.size;
        let r = 8.314462618;
        let mut theta = self.theta.clone();
        for (i, trap_layer) in self.trap_populations.iter().enumerate().take(n) {
            for (j, trap) in trap_layer.iter().enumerate() {
                theta[i][j] = (self.theta[i][j] + dt * trap.capture_rate * self.c_l[i])
                    / (1.0 + dt * (trap.capture_rate * self.c_l[i] + trap.release_rate));
            }
        }
        let mut next = self.c_l.clone();
        for i in 1..n.saturating_sub(1) {
            let material = &self.profiles[i];
            let stress_gradient = (self.hydrostatic_stress[i + 1] - self.hydrostatic_stress[i - 1])
                / (2.0 * self.grid.dx);
            let diffusion = material.diffusion_coeff
                * (self.c_l[i + 1] - 2.0 * self.c_l[i] + self.c_l[i - 1])
                / self.grid.dx.powi(2);
            let drift = material.diffusion_coeff
                * self.c_l[i]
                * material.partial_molar_volume
                * stress_gradient
                / (r * self.temperature);
            let trapping: f64 = self.trap_populations[i]
                .iter()
                .enumerate()
                .map(|(j, trap)| trap.density * (theta[i][j] - self.theta[i][j]) / dt)
                .sum();
            next[i] += dt * (diffusion - drift - trapping);
        }
        self.c_l = next;
        self.theta = theta;
    }

    /// Return the lattice-concentration partition factor across an interface.
    pub fn chemical_potential_partition(&self, left: usize, right: usize) -> f64 {
        let a = &self.profiles[left];
        let b = &self.profiles[right];
        (b.solubility / a.solubility)
            * ((b.partial_molar_volume * self.hydrostatic_stress[right]
                - a.partial_molar_volume * self.hydrostatic_stress[left])
                / (r_const() * self.temperature))
                .exp()
    }
}

/// Maximum physical atomic loading ratio `x_max` (D/M) before rupture of the
/// Pd0.9132Ir0.0868D_x active layer (cf3 spec §1).
pub const D_PD_MAX_PHYSICAL_LOADING_CAP: f64 = 0.9450;
/// Nominal operational loading of the active alloy, `x0 = 0.9132`.
pub const D_PD_SIMULATION_UPPER_BOUND: f64 = 0.9132;
/// Hydrostatic yield strength at 623.15 K [MPa]; exceeding `x_max` requires
/// stress beyond this limit.
pub const SIGMA_Y_HT_MPA: f64 = 155.0;

/// Soret stress-coupled diffusion flux
/// `J = −D_eff (∇C_L − (C_L V_H / (R T)) ∇σ_h)`.
///
/// `grad_c` is the lattice concentration gradient and `grad_sigma_h` the
/// hydrostatic stress gradient, in consistent units.
pub fn compute_soret_flux(c_l: f64, grad_c: f64, grad_sigma_h: f64, temp: f64, d_eff: f64) -> f64 {
    let v_h = 1.7e-6; // m^3/mol
    let r_gas = 8.314; // J/(mol K)
    -d_eff * (grad_c - (c_l * v_h / (r_gas * temp)) * grad_sigma_h)
}

/// Enforces the physical loading cap on the atomic ratio D/Pd.
///
/// Above `x_max = 0.9450` the required hydrostatic stress exceeds the 155 MPa
/// high-temperature yield strength, so reaching it mechanically is
/// impossible; this is a hard assertion. Ratios at or below `x_max` are
/// clamped to the nominal operational loading `0.9132`.
///
/// # Panics
/// If `loading_ratio > 0.9450` while `sigma_h <= 155.0` MPa.
pub fn enforce_loading_cap(loading_ratio: f64, sigma_h: f64) -> f64 {
    if loading_ratio > D_PD_MAX_PHYSICAL_LOADING_CAP {
        assert!(
            sigma_h > SIGMA_Y_HT_MPA,
            "Loading ratio > 0.9450 requires stress exceeding yield strength (155 MPa)"
        );
        return D_PD_MAX_PHYSICAL_LOADING_CAP;
    }
    loading_ratio.min(D_PD_SIMULATION_UPPER_BOUND)
}

fn r_const() -> f64 {
    8.314462618
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soret_flux_reduces_to_fickian_without_stress_gradient() {
        let j = compute_soret_flux(1.0, 5.0, 0.0, 623.15, 1e-11);
        assert!((j - -5.0e-11).abs() < 1e-20);
    }

    #[test]
    fn soret_flux_includes_stress_drift_term() {
        // J = -D (∇c − c·V_H/(RT)·∇σ) — stress term raises the flux here.
        let j = compute_soret_flux(1.0, 0.0, 1.0e9, 623.15, 1e-11);
        let expect = 1e-11 * (1.7e-6 / (8.314 * 623.15)) * 1.0e9;
        assert!((j - expect).abs() < 1e-6 * expect.abs().max(1e-20));
    }

    #[test]
    fn loading_cap_clamps_to_operational_bound() {
        assert_eq!(enforce_loading_cap(0.92, 100.0), 0.9132);
        assert_eq!(enforce_loading_cap(0.5, 100.0), 0.5);
    }

    #[test]
    fn loading_cap_above_max_clamps_when_stress_exceeds_yield() {
        assert_eq!(enforce_loading_cap(0.96, 1718.36), 0.9450);
    }

    #[test]
    #[should_panic(expected = "requires stress exceeding yield strength")]
    fn loading_cap_above_max_panics_below_yield_stress() {
        let _ = enforce_loading_cap(0.96, 100.0);
    }
}
