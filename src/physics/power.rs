// file: src/physics/power.rs

#[derive(Debug, Clone, Copy)]
pub struct PowerLedger {
    pub p_fusion: f64,
    pub p_optical_absorbed: f64,
    pub eta_teg: f64,
    pub p_parasitic_reported: f64,
    pub p_pump: f64,
    pub q_ref: f64,
    pub cop_ref: f64,
}

impl PowerLedger {
    pub fn compute_thermal_power(&self) -> f64 {
        self.p_fusion + self.p_optical_absorbed
    }

    pub fn compute_teg_electrical_power(&self) -> f64 {
        self.compute_thermal_power() * self.eta_teg
    }

    /// Computes the complete system net power output, resolving the pump gap
    pub fn compute_complete_net_power(&self) -> (f64, f64) {
        let p_thermal = self.compute_thermal_power();
        let p_teg_elec = p_thermal * self.eta_teg;

        let p_parasitic_complete = self.p_parasitic_reported + self.p_pump;
        let p_net_complete = p_teg_elec - p_parasitic_complete;

        (p_parasitic_complete, p_net_complete)
    }

    pub fn compute_compressor_duty(&self) -> f64 {
        self.q_ref / self.cop_ref
    }

    pub fn compute_teg_rejection_load(&self) -> f64 {
        self.compute_thermal_power() - self.compute_teg_electrical_power()
    }
}
