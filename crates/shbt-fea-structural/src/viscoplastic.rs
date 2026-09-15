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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_constants_match_update8_table() {
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
}
