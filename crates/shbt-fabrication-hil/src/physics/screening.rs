use rug::Float;

/// High-precision Thomas-Fermi screening and d-d Sommerfeld factors.
pub struct ScreeningKernel {
    pub density_of_states_ef: Float,
    pub atomic_density: Float,
    pub temperature: Float,
    pub screening_potential: Float,
}

impl ScreeningKernel {
    pub fn new(g_ef: Float, n_atom: Float, temp: Float) -> Self {
        let precision = g_ef.prec();
        let mut kernel = Self {
            density_of_states_ef: g_ef,
            atomic_density: n_atom,
            temperature: temp,
            screening_potential: Float::with_val(precision, 0),
        };
        kernel.compute_screening_potential();
        kernel
    }

    fn compute_screening_potential(&mut self) {
        let prec = self.density_of_states_ef.prec();
        let epsilon_0 = Float::with_val(prec, 8.8541878128e-12);
        let charge = Float::with_val(prec, 1.602176634e-19);
        let ev_to_joule = charge.clone();
        let density = Float::with_val(prec, &self.density_of_states_ef * &self.atomic_density) / &ev_to_joule;
        let lambda = (&epsilon_0 / (charge.clone() * charge.clone() * density)).sqrt();
        let pi = Float::with_val(prec, std::f64::consts::PI);
        let denominator = Float::with_val(prec, 4) * pi * epsilon_0 * lambda;
        self.screening_potential = (charge.clone() * charge / denominator) / ev_to_joule;
    }

    pub fn calculate_effective_sommerfeld(&self, energy_ev: &Float) -> Float {
        let prec = energy_ev.prec();
        let energy_kev = energy_ev / Float::with_val(prec, 1000);
        let two_pi_eta_0 = Float::with_val(prec, 31.28) / energy_kev.sqrt();
        let eta_0 = two_pi_eta_0 / Float::with_val(prec, 2.0 * std::f64::consts::PI);
        let denominator = Float::with_val(prec, energy_ev + &self.screening_potential);
        let ratio = energy_ev / denominator;
        eta_0 * ratio.sqrt()
    }

    pub fn calculate_enhancement_factor(&self, energy_ev: &Float) -> Float {
        let prec = energy_ev.prec();
        let energy_kev = energy_ev / Float::with_val(prec, 1000);
        let unscreened = Float::with_val(prec, 31.28) / energy_kev.sqrt();
        let eta = self.calculate_effective_sommerfeld(energy_ev);
        let screened = Float::with_val(prec, 2.0 * std::f64::consts::PI) * eta;
        (unscreened - screened).exp()
    }
}