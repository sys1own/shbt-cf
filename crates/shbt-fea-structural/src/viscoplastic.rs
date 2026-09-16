//! Non-isothermal Chaboche state update and Somigliana kernels.

use std::f64::consts::PI;

/// Material constants used by the Update-8 continuum benchmark at 623.15 K.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChabocheMaterial {
    /// Young's modulus [Pa].
    pub youngs_modulus: f64,
    /// Poisson ratio.
    pub poisson: f64,
    /// Isotropic thermal expansion [1/K].
    pub thermal_expansion: f64,
    /// Initial yield stress [Pa].
    pub yield_stress: f64,
    /// Overstress scale [Pa].
    pub viscosity: f64,
    /// Perzyna exponent.
    pub exponent: f64,
    /// Isotropic saturation [Pa].
    pub isotropic_saturation: f64,
    /// Isotropic hardening rate.
    pub isotropic_rate: f64,
    /// Kinematic hardening modulus [Pa].
    pub kinematic_modulus: f64,
    /// Dynamic recovery rate.
    pub recovery: f64,
}

impl ChabocheMaterial {
    /// Inconel 718 at 350 °C.
    pub const INCONEL_718: Self = Self {
        youngs_modulus: 172e9,
        poisson: 0.31,
        thermal_expansion: 14.2e-6,
        yield_stress: 820e6,
        viscosity: 20e6,
        exponent: 5.0,
        isotropic_saturation: 150e6,
        isotropic_rate: 8.0,
        kinematic_modulus: 40e9,
        recovery: 12.0,
    };
    /// Alumina support at 350 °C.
    pub const ALUMINA: Self = Self {
        youngs_modulus: 340e9,
        poisson: 0.22,
        thermal_expansion: 8.1e-6,
        yield_stress: 2100e6,
        viscosity: 50e6,
        exponent: 5.0,
        isotropic_saturation: 250e6,
        isotropic_rate: 8.0,
        kinematic_modulus: 20e9,
        recovery: 8.0,
    };
    /// TLP-Ag interface at 350 °C.
    pub const TLP_AG: Self = Self {
        youngs_modulus: 61e9,
        poisson: 0.37,
        thermal_expansion: 19.5e-6,
        yield_stress: 45e6,
        viscosity: 5e6,
        exponent: 4.0,
        isotropic_saturation: 30e6,
        isotropic_rate: 6.0,
        kinematic_modulus: 8e9,
        recovery: 4.0,
    };

    fn shear(self) -> f64 {
        self.youngs_modulus / (2.0 * (1.0 + self.poisson))
    }
    fn bulk(self) -> f64 {
        self.youngs_modulus / (3.0 * (1.0 - 2.0 * self.poisson))
    }
}

/// Symmetric tensor represented in row-major 3 x 3 form.
pub type Tensor = [[f64; 3]; 3];

/// Chaboche internal state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViscoplasticState {
    /// Cauchy stress [Pa].
    pub stress: Tensor,
    /// Back stress [Pa].
    pub back_stress: Tensor,
    /// Isotropic hardening stress [Pa].
    pub isotropic_hardening: f64,
    /// Accumulated plastic strain.
    pub accumulated_plastic_strain: f64,
}

fn deviatoric(t: Tensor) -> Tensor {
    let trace = (t[0][0] + t[1][1] + t[2][2]) / 3.0;
    let mut result = t;
    for (i, row) in result.iter_mut().enumerate() {
        row[i] -= trace;
    }
    result
}

fn inner(a: Tensor, b: Tensor) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x.iter().zip(y.iter()).map(|(u, v)| u * v).sum::<f64>())
        .sum()
}

/// Advance one explicit thermoviscoplastic step.
pub fn update_state(
    state: &mut ViscoplasticState,
    total_strain_rate: Tensor,
    temperature_rate: f64,
    dt: f64,
    material: ChabocheMaterial,
) -> f64 {
    assert!(dt >= 0.0 && temperature_rate.is_finite());
    let thermal_rate = material.thermal_expansion * temperature_rate;
    let mut mechanical = total_strain_rate;
    for (i, row) in mechanical.iter_mut().enumerate() {
        row[i] -= thermal_rate;
    }
    let elastic_trace = mechanical[0][0] + mechanical[1][1] + mechanical[2][2];
    let trial = state.stress;
    let trial_dev = deviatoric(trial);
    let effective = {
        let mut d = trial_dev;
        for (i, row) in d.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                *value -= state.back_stress[i][j];
            }
        }
        (1.5 * inner(d, d)).sqrt()
    };
    let overstress = (effective - state.isotropic_hardening - material.yield_stress).max(0.0);
    let plastic_rate = (overstress / material.viscosity).powf(material.exponent);
    let direction_norm = effective.max(f64::MIN_POSITIVE);
    let mut flow = [[0.0; 3]; 3];
    for (i, row) in flow.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            *value =
                1.5 * plastic_rate * (trial_dev[i][j] - state.back_stress[i][j]) / direction_norm;
        }
    }
    let mut stress_rate = [[0.0; 3]; 3];
    for (i, row) in stress_rate.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            *value = 2.0 * material.shear() * (mechanical[i][j] - flow[i][j])
                + if i == j {
                    material.bulk() * elastic_trace
                } else {
                    0.0
                };
        }
    }
    for (i, row) in state.stress.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            *value += stress_rate[i][j] * dt;
            state.back_stress[i][j] += (material.kinematic_modulus * flow[i][j]
                - material.recovery * state.back_stress[i][j] * plastic_rate)
                * dt;
        }
    }
    state.isotropic_hardening += material.isotropic_rate
        * (material.isotropic_saturation - state.isotropic_hardening)
        * plastic_rate
        * dt;
    state.accumulated_plastic_strain += plastic_rate * dt;
    plastic_rate
}

/// Kelvin displacement kernel `U*` for an isotropic infinite solid.
pub fn somigliana_displacement_kernel(
    x_minus_y: [f64; 3],
    poisson: f64,
    shear: f64,
) -> [[f64; 3]; 3] {
    let r2 = inner_vec(x_minus_y, x_minus_y);
    let r = r2.sqrt().max(f64::MIN_POSITIVE);
    let mut result = [[0.0; 3]; 3];
    for (i, row) in result.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            *value = ((3.0 - 4.0 * poisson) * f64::from(i == j)
                + x_minus_y[i] * x_minus_y[j] / r2.max(f64::MIN_POSITIVE))
                / (16.0 * PI * (1.0 - poisson) * shear * r);
        }
    }
    result
}

fn inner_vec(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Two-term non-isothermal Chaboche viscoplastic model for
/// `Pd_{0.9132}Ir_{0.0868}` on a SiC substrate (update-11.1 Task 3).
///
/// Perzyna flow: `ε̇^vp_ij = (3/2) ṗ (s_ij − X'_ij)/J2(σ − X)` with
/// `ṗ = ⟨f/K⟩^n`, `f = J2(σ − X) − R − σ_y(T)`, `R = Q(1 − e^{−bp})`.
/// Dual kinematic backstress `X = X1 + X2` evolves as
/// `Ẋ_k = (2/3)C_k(T) ε̇^vp − γ_k X_k ṗ + (1/C_k)(∂C_k/∂T) Ṫ X_k`.
///
/// All stresses in MPa.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChabocheViscoplasticModel {
    /// Viscosity reference stress `K` [MPa·s^(1/n)].
    pub k_visco: f64,
    /// Perzyna exponent `n`.
    pub n_visco: f64,
    /// Isotropic saturation stress `Q` [MPa].
    pub q_iso: f64,
    /// Isotropic rate `b`.
    pub b_iso: f64,
    /// Dynamic recovery `γ_1` of the first backstress term.
    pub gamma1: f64,
    /// Dynamic recovery `γ_2` of the second backstress term.
    pub gamma2: f64,
    /// First kinematic backstress component `X_1` [MPa].
    pub back_stress_1: f64,
    /// Second kinematic backstress component `X_2` [MPa].
    pub back_stress_2: f64,
}

/// Single-cycle initial plastic strain increment bound.
pub const DP_CYCLE_BOUND: f64 = 8.952e-4;

impl ChabocheViscoplasticModel {
    /// Pd–Ir film constants: `K = 60 MPa·s^{1/n}`, `n = 4`, `Q = 40 MPa`,
    /// `b = 10`, `γ_1 = 500`, `γ_2 = 80`.
    pub fn pd_ir() -> Self {
        Self {
            k_visco: 60.0,
            n_visco: 4.0,
            q_iso: 40.0,
            b_iso: 10.0,
            gamma1: 500.0,
            gamma2: 80.0,
            back_stress_1: 0.0,
            back_stress_2: 0.0,
        }
    }

    /// Temperature-dependent yield `σ_y(T) = 220 − 0.2(T − 298.15)` [MPa].
    pub fn yield_strength(&self, temp_k: f64) -> f64 {
        220.0 - 0.2 * (temp_k - 298.15)
    }

    /// `C_1(T) = 50.0e3 − 61.54(T − 298.15)` [MPa].
    pub fn c1_modulus(&self, temp_k: f64) -> f64 {
        50.0e3 - 61.54 * (temp_k - 298.15)
    }

    /// `C_2(T) = 15.0e3 − 21.54(T − 298.15)` [MPa].
    pub fn c2_modulus(&self, temp_k: f64) -> f64 {
        15.0e3 - 21.54 * (temp_k - 298.15)
    }

    /// Isotropic hardening `R(p) = Q(1 − e^{−b p})` [MPa].
    pub fn isotropic_hardening(&self, p: f64) -> f64 {
        self.q_iso * (1.0 - (-self.b_iso * p).exp())
    }

    /// Total backstress `X = X_1 + X_2` [MPa].
    pub fn back_stress(&self) -> f64 {
        self.back_stress_1 + self.back_stress_2
    }

    /// Advances the dual backstress terms by `dt` under viscoplastic strain
    /// rate `deps_vp`, accumulated plastic rate `dp`, and temperature rate
    /// `dtemp`, at temperature `temp_k`.
    pub fn evolve_backstress(&mut self, deps_vp: f64, dp: f64, dtemp: f64, temp_k: f64, dt: f64) {
        let c1 = self.c1_modulus(temp_k);
        let c2 = self.c2_modulus(temp_k);
        self.back_stress_1 += ((2.0 / 3.0) * c1 * deps_vp - self.gamma1 * self.back_stress_1 * dp
            + (-61.54 / c1) * dtemp * self.back_stress_1)
            * dt;
        self.back_stress_2 += ((2.0 / 3.0) * c2 * deps_vp - self.gamma2 * self.back_stress_2 * dp
            + (-21.54 / c2) * dtemp * self.back_stress_2)
            * dt;
    }

    /// Perzyna accumulated-plastic-strain rate `ṗ = ⟨f/K⟩^n`.
    pub fn plastic_rate(&self, overstress: f64) -> f64 {
        (overstress / self.k_visco).max(0.0).powf(self.n_visco)
    }

    /// Enforces the single-cycle plastic-increment bound.
    ///
    /// # Panics
    /// If `dp_cycle > 8.952e-4`.
    pub fn assert_initial_increment(dp_cycle: f64) {
        assert!(
            dp_cycle <= DP_CYCLE_BOUND,
            "Initial plastic strain increment exceeds single-cycle bound"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_constants_match_spec_table() {
        assert_eq!(ChabocheMaterial::INCONEL_718.youngs_modulus, 172e9);
        assert_eq!(ChabocheMaterial::TLP_AG.yield_stress, 45e6);
    }

    #[test]
    fn elastic_step_is_finite_and_kernel_is_symmetric() {
        let mut state = ViscoplasticState {
            stress: [[0.0; 3]; 3],
            back_stress: [[0.0; 3]; 3],
            isotropic_hardening: 0.0,
            accumulated_plastic_strain: 0.0,
        };
        let rate = update_state(
            &mut state,
            [[1e-6, 0.0, 0.0], [0.0, -0.5e-6, 0.0], [0.0, 0.0, -0.5e-6]],
            0.0,
            1e-3,
            ChabocheMaterial::INCONEL_718,
        );
        assert!(rate.is_finite() && state.stress[0][0].is_finite());
        let kernel = somigliana_displacement_kernel([1.0, 0.0, 0.0], 0.3, 1.0);
        assert_eq!(kernel[0][1], kernel[1][0]);
    }

    #[test]
    fn pd_ir_model_temperature_functions_match_spec() {
        let m = ChabocheViscoplasticModel::pd_ir();
        assert!((m.yield_strength(298.15) - 220.0).abs() < 1e-9);
        assert!((m.yield_strength(623.15) - 155.0).abs() < 1e-9);
        assert!((m.c1_modulus(298.15) - 50.0e3).abs() < 1e-9);
        assert!((m.c2_modulus(623.15) - 15.0e3 + 21.54 * 325.0).abs() < 1e-9);
        assert_eq!(m.back_stress(), 0.0);
    }

    #[test]
    fn initial_increment_bound_enforced() {
        ChabocheViscoplasticModel::assert_initial_increment(8.952e-4);
    }

    #[test]
    #[should_panic(expected = "single-cycle bound")]
    fn initial_increment_over_bound_panics() {
        ChabocheViscoplasticModel::assert_initial_increment(1.0e-3);
    }

    #[test]
    fn dual_backstress_evolves() {
        let mut m = ChabocheViscoplasticModel::pd_ir();
        m.evolve_backstress(1e-5, 1e-6, 0.0, 623.15, 1.0);
        assert!(m.back_stress_1 > 0.0 && m.back_stress_2 > 0.0);
        assert!(m.back_stress_1 > m.back_stress_2); // C1 > C2
    }
}
