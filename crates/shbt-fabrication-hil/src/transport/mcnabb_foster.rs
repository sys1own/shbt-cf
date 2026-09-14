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
        for i in 0..n { for j in 0..self.trap_populations[i].len() {
            let trap = &self.trap_populations[i][j];
            theta[i][j] = (self.theta[i][j] + dt * trap.capture_rate * self.c_l[i]) /
                (1.0 + dt * (trap.capture_rate * self.c_l[i] + trap.release_rate));
        }}
        let mut next = self.c_l.clone();
        for i in 1..n.saturating_sub(1) {
            let material = &self.profiles[i];
            let stress_gradient = (self.hydrostatic_stress[i + 1] - self.hydrostatic_stress[i - 1]) / (2.0 * self.grid.dx);
            let diffusion = material.diffusion_coeff * (self.c_l[i + 1] - 2.0 * self.c_l[i] + self.c_l[i - 1]) / self.grid.dx.powi(2);
            let drift = material.diffusion_coeff * self.c_l[i] * material.partial_molar_volume * stress_gradient / (r * self.temperature);
            let trapping: f64 = self.trap_populations[i].iter().enumerate().map(|(j, trap)| trap.density * (theta[i][j] - self.theta[i][j]) / dt).sum();
            next[i] += dt * (diffusion - drift - trapping);
        }
        self.c_l = next;
        self.theta = theta;
    }

    /// Return the lattice-concentration partition factor across an interface.
    pub fn chemical_potential_partition(&self, left: usize, right: usize) -> f64 {
        let a = &self.profiles[left]; let b = &self.profiles[right];
        (b.solubility / a.solubility) * ((b.partial_molar_volume * self.hydrostatic_stress[right] - a.partial_molar_volume * self.hydrostatic_stress[left]) / (r_const() * self.temperature)).exp()
    }
}

fn r_const() -> f64 { 8.314462618 }