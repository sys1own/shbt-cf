#[derive(Clone, Debug)]
pub struct TransportGrid { pub dx: f64, pub size: usize, pub interface_boundaries: Vec<usize> }

#[derive(Clone, Debug)]
pub struct MaterialProfile { pub diffusion_coeff: f64, pub solubility: f64, pub partial_molar_volume: f64 }

#[derive(Clone, Debug)]
pub struct TrapState { pub density: f64, pub capture_rate: f64, pub release_rate: f64 }

pub struct McNabbFosterSolver {
    pub grid: TransportGrid,
    pub profiles: Vec<MaterialProfile>,
    pub trap_populations: Vec<Vec<TrapState>>,
    pub c_l: Vec<f64>,
    pub theta: Vec<Vec<f64>>,
    pub hydrostatic_stress: Vec<f64>,
    pub temperature: f64,
}

impl McNabbFosterSolver {
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

    pub fn chemical_potential_partition(&self, left: usize, right: usize) -> f64 {
        let a = &self.profiles[left]; let b = &self.profiles[right];
        (b.solubility / a.solubility) * ((b.partial_molar_volume * self.hydrostatic_stress[right] - a.partial_molar_volume * self.hydrostatic_stress[left]) / (r_const() * self.temperature)).exp()
    }
}

fn r_const() -> f64 { 8.314462618 }