//! Thermodynamic TLP phase evolution and dislocation-climb creep.

const K_B: f64 = 1.380_649e-23;

/// One periodic one-dimensional Cahn-Hilliard/Allen-Cahn state.
#[derive(Clone, Debug, PartialEq)]
pub struct PhaseFieldState {
    /// Conserved solute fraction at cell centres.
    pub concentration: Vec<f64>,
    /// Non-conserved phase order parameters.
    pub order_parameters: Vec<Vec<f64>>,
    /// Cell spacing [m].
    pub spacing: f64,
}

impl PhaseFieldState {
    fn laplacian(field: &[f64], spacing: f64) -> Vec<f64> {
        let n = field.len();
        (0..n)
            .map(|i| {
                (field[(i + n - 1) % n] - 2.0 * field[i] + field[(i + 1) % n]) / (spacing * spacing)
            })
            .collect()
    }

    /// Advances coupled Cahn-Hilliard and Allen-Cahn equations by one explicit step.
    pub fn step(&mut self, dt: f64, mobility: f64, gradient_energy: f64, barrier: f64) {
        let lap_c = Self::laplacian(&self.concentration, self.spacing);
        let chemical: Vec<f64> = self
            .concentration
            .iter()
            .zip(&lap_c)
            .map(|(&c, &lap)| c * c * c - c - gradient_energy * lap)
            .collect();
        let lap_mu = Self::laplacian(&chemical, self.spacing);
        for (c, &lap) in self.concentration.iter_mut().zip(&lap_mu) {
            *c = (*c + dt * mobility * lap).clamp(0.0, 1.0);
        }
        for order in &mut self.order_parameters {
            let lap = Self::laplacian(order, self.spacing);
            for (eta, (&c, &lap_eta)) in order.iter_mut().zip(self.concentration.iter().zip(&lap)) {
                let derivative =
                    barrier * 2.0 * *eta * (1.0 - *eta) * (1.0 - 2.0 * *eta) + 2.0 * (*eta - c);
                *eta = (*eta + dt * (lap_eta - derivative)).clamp(0.0, 1.0);
            }
        }
    }
}

/// Inputs to the Weertman vacancy-diffusion climb model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeertmanInputs {
    /// Equivalent stress [Pa].
    pub equivalent_stress: f64,
    /// Hydrostatic stress [Pa].
    pub hydrostatic_stress: f64,
    /// Shear modulus [Pa].
    pub shear_modulus: f64,
    /// Burgers vector [m].
    pub burgers_vector: f64,
    /// Vacancy diffusivity [m²/s].
    pub vacancy_diffusivity: f64,
    /// Equilibrium vacancy fraction.
    pub vacancy_fraction: f64,
    /// Atomic volume [m³].
    pub atomic_volume: f64,
    /// Temperature [K].
    pub temperature: f64,
    /// Dislocation outer/core radii [m].
    pub outer_radius: f64,
    /// Dislocation core radius [m].
    pub core_radius: f64,
}

/// Orowan strain rate from vacancy diffusion assisted dislocation climb [1/s].
pub fn weertman_creep_rate(input: WeertmanInputs) -> f64 {
    let stress_scale = input.equivalent_stress / (input.shear_modulus * input.burgers_vector);
    let density = stress_scale * stress_scale;
    let core = input.vacancy_fraction
        * (input.hydrostatic_stress * input.atomic_volume / (K_B * input.temperature)).exp();
    let flux =
        2.0 * std::f64::consts::PI * input.vacancy_diffusivity * (input.vacancy_fraction - core)
            / (input.outer_radius / input.core_radius).ln();
    density * input.burgers_vector * flux.abs()
}

/// Leak-before-break margin evaluation (update-11.1 Task 3).
///
/// `2a_c = (2/π)(K_IH/(Y·σ_m))²` — critical through-wall crack length [mm];
/// `M_a = 2a_c / a_leak` — LBB margin; compliant iff `M_a ≥ 2.0`.
///
/// `k_ih` in MPa·√m, `sigma_m` in MPa, `a_leak_mm` in mm.
/// Returns `(two_a_c_mm, margin, is_compliant)`.
pub fn evaluate_lbb_margin(
    k_ih: f64,
    sigma_m: f64,
    y_factor: f64,
    a_leak_mm: f64,
) -> (f64, f64, bool) {
    let pi = std::f64::consts::PI;
    let a_c_m = (1.0 / pi) * (k_ih / (y_factor * sigma_m)).powi(2);
    let two_a_c_mm = 2.0 * a_c_m * 1000.0;
    let margin = two_a_c_mm / a_leak_mm;
    let is_compliant = margin >= 2.0;
    (two_a_c_mm, margin, is_compliant)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn phase_step_preserves_periodic_mean_to_roundoff() {
        let mut state = PhaseFieldState {
            concentration: vec![0.3, 0.4, 0.5, 0.4],
            order_parameters: vec![vec![0.2; 4]],
            spacing: 1.0,
        };
        let before = state.concentration.iter().sum::<f64>();
        state.step(1e-4, 1e-3, 0.1, 1.0);
        assert!((state.concentration.iter().sum::<f64>() - before).abs() < 1e-12);
    }

    #[test]
    fn lbb_margin_matches_spec_validation() {
        let (two_a_c, margin, compliant) = evaluate_lbb_margin(45.0, 180.0, 1.12, 12.0);
        assert!(
            (two_a_c - 31.6).abs() < 0.2,
            "Critical crack length mismatch"
        );
        assert!(margin >= 2.6, "LBB margin safety check failed");
        assert!(compliant);
    }

    #[test]
    fn lbb_margin_flags_noncompliant() {
        let (_, margin, compliant) = evaluate_lbb_margin(20.0, 300.0, 1.12, 12.0);
        assert!(margin < 2.0 && !compliant);
    }
}
