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

/// Single-cycle plastic strain increment bound: the cf3 FEA strain extraction
/// at the TLP bondline hotspot gives \Delta\epsilon_p = 0.00112 per thermal
/// reversal (298.15 K <-> 623.15 K).
pub const DP_CYCLE_BOUND: f64 = 1.12e-3;

/// Thermal cycle endpoints for the 5-layer stack (cf3 spec §2.1).
pub const T_CYCLE_MIN_K: f64 = 298.15;
/// Upper endpoint of the thermal cycle.
pub const T_CYCLE_MAX_K: f64 = 623.15;

impl ChabocheViscoplasticModel {
    /// Pd0.9132Ir0.0868 film constants (cf3 spec §2.1): `K = 60 MPa·s^{1/n}`,
    /// `n = 4`, `R_inf = 85.0 MPa`, `b = 12.46`, `γ_1 = 410`, `γ_2 = 62`
    /// at the 298.15 K endpoint.
    pub fn pd_ir() -> Self {
        Self {
            k_visco: 60.0,
            n_visco: 4.0,
            q_iso: 85.0,
            b_iso: 12.46,
            gamma1: 410.0,
            gamma2: 62.0,
            back_stress_1: 0.0,
            back_stress_2: 0.0,
        }
    }

    /// Temperature-dependent yield `σ_y(T) = 215 − 0.22462(T − 298.15)` [MPa],
    /// interpolating 215.0 MPa at 298.15 K to 142.0 MPa at 623.15 K.
    pub fn yield_strength(&self, temp_k: f64) -> f64 {
        215.0 - 73.0 * (temp_k - 298.15) / 325.0
    }

    /// `C_1(T) = 45.2e3 − 43.692(T − 298.15)` [MPa], interpolating
    /// 45.2 GPa at 298.15 K to 31.0 GPa at 623.15 K.
    pub fn c1_modulus(&self, temp_k: f64) -> f64 {
        45.2e3 - 14.2e3 * (temp_k - 298.15) / 325.0
    }

    /// `C_2(T) = 8.5e3 − 10.154(T − 298.15)` [MPa], interpolating
    /// 8.5 GPa at 298.15 K to 5.2 GPa at 623.15 K.
    pub fn c2_modulus(&self, temp_k: f64) -> f64 {
        8.5e3 - 3.3e3 * (temp_k - 298.15) / 325.0
    }

    /// `γ_1(T)`: 410 at 298.15 K → 380 at 623.15 K.
    pub fn gamma1_at(&self, temp_k: f64) -> f64 {
        410.0 - 30.0 * (temp_k - 298.15) / 325.0
    }

    /// `γ_2(T)`: 62 at 298.15 K → 48 at 623.15 K.
    pub fn gamma2_at(&self, temp_k: f64) -> f64 {
        62.0 - 14.0 * (temp_k - 298.15) / 325.0
    }

    /// `R_inf(T)` [MPa]: 85.0 at 298.15 K → 52.0 at 623.15 K.
    pub fn q_iso_at(&self, temp_k: f64) -> f64 {
        85.0 - 33.0 * (temp_k - 298.15) / 325.0
    }

    /// `b(T)`: 12.46 at 298.15 K → 10.1 at 623.15 K.
    pub fn b_iso_at(&self, temp_k: f64) -> f64 {
        12.46 - 2.36 * (temp_k - 298.15) / 325.0
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
        let gamma1 = self.gamma1_at(temp_k);
        let gamma2 = self.gamma2_at(temp_k);
        self.back_stress_1 += ((2.0 / 3.0) * c1 * deps_vp - gamma1 * self.back_stress_1 * dp
            + (-14.2e3 / 325.0 / c1) * dtemp * self.back_stress_1)
            * dt;
        self.back_stress_2 += ((2.0 / 3.0) * c2 * deps_vp - gamma2 * self.back_stress_2 * dp
            + (-3.3e3 / 325.0 / c2) * dtemp * self.back_stress_2)
            * dt;
    }

    /// Perzyna accumulated-plastic-strain rate `ṗ = ⟨f/K⟩^n`.
    pub fn plastic_rate(&self, overstress: f64) -> f64 {
        (overstress / self.k_visco).max(0.0).powf(self.n_visco)
    }

    /// Enforces the single-cycle plastic-increment bound.
    ///
    /// # Panics
    /// If `dp_cycle > 1.12e-3`.
    pub fn assert_initial_increment(dp_cycle: f64) {
        assert!(
            dp_cycle <= DP_CYCLE_BOUND,
            "Initial plastic strain increment exceeds single-cycle bound"
        );
    }
}

// ---------------------------------------------------------------------------
// 5-layer reactor stack constitutive table (cf3 spec §2.1), VCCT joint
// delamination (§2.2) and Coffin–Manson–Morrow strain-life fatigue (§2.3).
// ---------------------------------------------------------------------------

/// One metallic layer of the 5-layer reactor stack at one thermal endpoint.
/// CVD Diamond is purely thermoelastic below 1000 K and carries no Chaboche
/// constants (`None`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StackLayerChaboche {
    /// Young's modulus [GPa].
    pub e_gpa: f64,
    /// Poisson ratio.
    pub poisson: f64,
    /// Coefficient of thermal expansion [1/K].
    pub cte: f64,
    /// Initial yield stress [MPa].
    pub yield_mpa: f64,
    /// First kinematic hardening modulus `C_1` [GPa].
    pub c1_gpa: f64,
    /// First dynamic recovery rate `γ_1`.
    pub gamma1: f64,
    /// Second kinematic hardening modulus `C_2` [GPa].
    pub c2_gpa: f64,
    /// Second dynamic recovery rate `γ_2`.
    pub gamma2: f64,
    /// Saturated isotropic hardening stress `R_inf` [MPa].
    pub r_inf_mpa: f64,
    /// Isotropic hardening stabilization rate `b`.
    pub b_iso: f64,
}

/// Layer identifiers for the mechanically coupled 5-layer stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StackLayer {
    /// Pd0.9132Ir0.0868 active target film (0.5 um).
    PdIr,
    /// Titanium adhesion/barrier interlayer (10 nm).
    Titanium,
    /// Ni-Cu-Sn transient liquid phase bondline (3.5 um).
    TlpBondline,
    /// OFHC copper cold plate (1.0 mm).
    OfhcCu,
}

impl StackLayer {
    /// Endpoint constants at `T_CYCLE_MIN_K` (298.15 K).
    pub const fn low(self) -> StackLayerChaboche {
        match self {
            Self::PdIr => StackLayerChaboche {
                e_gpa: 138.5,
                poisson: 0.38,
                cte: 11.8e-6,
                yield_mpa: 215.0,
                c1_gpa: 45.2,
                gamma1: 410.0,
                c2_gpa: 8.5,
                gamma2: 62.0,
                r_inf_mpa: 85.0,
                b_iso: 12.46,
            },
            Self::Titanium => StackLayerChaboche {
                e_gpa: 116.0,
                poisson: 0.32,
                cte: 8.6e-6,
                yield_mpa: 280.0,
                c1_gpa: 38.0,
                gamma1: 290.0,
                c2_gpa: 6.1,
                gamma2: 35.0,
                r_inf_mpa: 110.0,
                b_iso: 8.5,
            },
            Self::TlpBondline => StackLayerChaboche {
                e_gpa: 128.0,
                poisson: 0.31,
                cte: 16.4e-6,
                yield_mpa: 310.0,
                c1_gpa: 62.0,
                gamma1: 680.0,
                c2_gpa: 12.4,
                gamma2: 95.0,
                r_inf_mpa: 145.0,
                b_iso: 18.2,
            },
            Self::OfhcCu => StackLayerChaboche {
                e_gpa: 115.0,
                poisson: 0.34,
                cte: 16.8e-6,
                yield_mpa: 75.0,
                c1_gpa: 28.5,
                gamma1: 520.0,
                c2_gpa: 3.8,
                gamma2: 42.0,
                r_inf_mpa: 65.0,
                b_iso: 15.5,
            },
        }
    }

    /// Endpoint constants at `T_CYCLE_MAX_K` (623.15 K).
    pub const fn high(self) -> StackLayerChaboche {
        match self {
            Self::PdIr => StackLayerChaboche {
                e_gpa: 112.0,
                poisson: 0.40,
                cte: 13.2e-6,
                yield_mpa: 142.0,
                c1_gpa: 31.0,
                gamma1: 380.0,
                c2_gpa: 5.2,
                gamma2: 48.0,
                r_inf_mpa: 52.0,
                b_iso: 10.1,
            },
            Self::Titanium => StackLayerChaboche {
                e_gpa: 94.5,
                poisson: 0.34,
                cte: 9.8e-6,
                yield_mpa: 185.0,
                c1_gpa: 26.5,
                gamma1: 260.0,
                c2_gpa: 4.0,
                gamma2: 28.0,
                r_inf_mpa: 74.0,
                b_iso: 7.2,
            },
            Self::TlpBondline => StackLayerChaboche {
                e_gpa: 96.0,
                poisson: 0.33,
                cte: 18.1e-6,
                yield_mpa: 195.0,
                c1_gpa: 41.5,
                gamma1: 590.0,
                c2_gpa: 8.1,
                gamma2: 72.0,
                r_inf_mpa: 92.0,
                b_iso: 14.8,
            },
            Self::OfhcCu => StackLayerChaboche {
                e_gpa: 88.0,
                poisson: 0.36,
                cte: 18.6e-6,
                yield_mpa: 42.0,
                c1_gpa: 18.0,
                gamma1: 460.0,
                c2_gpa: 2.2,
                gamma2: 31.0,
                r_inf_mpa: 38.0,
                b_iso: 12.0,
            },
        }
    }

    /// Piecewise-linear interpolation of all constants between the thermal
    /// cycle endpoints (cf3 spec §2.1 temperature scaling).
    pub fn at(self, temp_k: f64) -> StackLayerChaboche {
        let lo = self.low();
        let hi = self.high();
        let t = ((temp_k - T_CYCLE_MIN_K) / (T_CYCLE_MAX_K - T_CYCLE_MIN_K)).clamp(0.0, 1.0);
        StackLayerChaboche {
            e_gpa: lo.e_gpa + t * (hi.e_gpa - lo.e_gpa),
            poisson: lo.poisson + t * (hi.poisson - lo.poisson),
            cte: lo.cte + t * (hi.cte - lo.cte),
            yield_mpa: lo.yield_mpa + t * (hi.yield_mpa - lo.yield_mpa),
            c1_gpa: lo.c1_gpa + t * (hi.c1_gpa - lo.c1_gpa),
            gamma1: lo.gamma1 + t * (hi.gamma1 - lo.gamma1),
            c2_gpa: lo.c2_gpa + t * (hi.c2_gpa - lo.c2_gpa),
            gamma2: lo.gamma2 + t * (hi.gamma2 - lo.gamma2),
            r_inf_mpa: lo.r_inf_mpa + t * (hi.r_inf_mpa - lo.r_inf_mpa),
            b_iso: lo.b_iso + t * (hi.b_iso - lo.b_iso),
        }
    }
}

/// Mixed-mode VCCT energy release rates at one crack-front node [J/m^2].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VcctReleaseRates {
    /// Mode I (opening).
    pub g_i: f64,
    /// Mode II (in-plane shear).
    pub g_ii: f64,
    /// Mode III (out-of-plane tear).
    pub g_iii: f64,
}

/// Critical strain energy release rates of the (Cu,Ni)6Sn5 / CVD-diamond
/// interface [J/m^2] (cf3 spec §2.2).
pub const G_IC: f64 = 25.0;
/// Mode II critical energy release rate [J/m^2].
pub const G_IIC: f64 = 65.0;
/// Mode III critical energy release rate [J/m^2].
pub const G_IIIC: f64 = 60.0;

/// Linear power-law mixed-mode delamination criterion:
/// `f_delam = G_I/G_IC + G_II/G_IIC + G_III/G_IIIC`; initiation at `>= 1.0`.
pub fn vcct_delamination_factor(g: VcctReleaseRates) -> f64 {
    g.g_i / G_IC + g.g_ii / G_IIC + g.g_iii / G_IIIC
}

/// VCCT nodal energy release rates for one crack-tip pair:
/// `G_I = Z_i (w_l - w_l')/(2 \Delta A)` etc. `forces` are `[X_i, Y_i, Z_i]`
/// [N], `displacements` the relative crack-surface opening `[du, dv, dw]` [m],
/// `delta_a` the element edge length and `delta_b` the crack-front thickness.
pub fn vcct_release_rates(
    forces: [f64; 3],
    displacements: [f64; 3],
    delta_a: f64,
    delta_b: f64,
) -> VcctReleaseRates {
    let inv = 1.0 / (2.0 * delta_a * delta_b);
    VcctReleaseRates {
        g_ii: forces[0] * displacements[0] * inv,
        g_iii: forces[1] * displacements[1] * inv,
        g_i: forces[2] * displacements[2] * inv,
    }
}

/// Peak VCCT release rates at the outer perimeter of the 3.5 um Ni-Cu-Sn TLP
/// bondline under the 623.15 K -> 298.15 K shock at 100 K/s (cf3 spec §2.2).
pub fn tlp_bondline_peak_release_rates() -> VcctReleaseRates {
    VcctReleaseRates {
        g_i: 4.12,
        g_ii: 18.45,
        g_iii: 2.10,
    }
}

/// Coffin–Manson–Morrow strain-life constants of the Ni-Cu-Sn TLP
/// bondline (cf3 spec §2.3): `σ_f' = 520 MPa`, `b = -0.095`,
/// `ε_f' = 0.380`, `c = -0.580`, `E = 112 GPa`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrainLife {
    /// Fatigue strength coefficient [MPa].
    pub sigma_f_prime_mpa: f64,
    /// Fatigue strength exponent.
    pub b: f64,
    /// Fatigue ductility coefficient.
    pub epsilon_f_prime: f64,
    /// Fatigue ductility exponent.
    pub c: f64,
    /// Temperature-averaged Young's modulus [MPa].
    pub e_mpa: f64,
}

/// TLP bondline strain-life constants (cf3 spec §2.3).
pub const TLP_STRAIN_LIFE: StrainLife = StrainLife {
    sigma_f_prime_mpa: 520.0,
    b: -0.095,
    epsilon_f_prime: 0.380,
    c: -0.580,
    e_mpa: 112_000.0,
};

/// Minimum procurement fatigue endurance `N_req` [cycles].
pub const FATIGUE_LIFE_REQUIRED: f64 = 52_400.0;

impl StrainLife {
    /// Total strain amplitude predicted at `reversals` (= 2 N_f):
    /// `Δε_t/2 = (σ_f' - σ_m)/E · (2N_f)^b + ε_f' · (2N_f)^c`.
    pub fn strain_amplitude(&self, reversals: f64, sigma_m_mpa: f64) -> f64 {
        (self.sigma_f_prime_mpa - sigma_m_mpa) / self.e_mpa * reversals.powf(self.b)
            + self.epsilon_f_prime * reversals.powf(self.c)
    }

    /// Solves the Coffin–Manson–Morrow strain-life equation for cycles
    /// to failure `N_f` by bisection on `2 N_f`.
    pub fn cycles_to_failure(&self, strain_amplitude: f64, sigma_m_mpa: f64) -> f64 {
        assert!(strain_amplitude > 0.0);
        let mut lo = 2.0f64;
        let mut hi = 1.0e12f64;
        for _ in 0..200 {
            let mid = (lo * hi).sqrt();
            if self.strain_amplitude(mid, sigma_m_mpa) > strain_amplitude {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (lo * hi).sqrt() / 2.0
    }
}

/// FEA-extracted critical hotspot strain state for the TLP bondline corner
/// under steady-state cycling 298.15 K <-> 623.15 K (cf3 spec §2.3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TlpHotspotStrain {
    /// Total strain amplitude `Δε_t/2`.
    pub total_strain_amp: f64,
    /// Plastic strain amplitude `Δε_p/2`.
    pub plastic_strain_amp: f64,
    /// Elastic strain amplitude `Δε_e/2`.
    pub elastic_strain_amp: f64,
    /// Mean tensile stress post-cooling [MPa].
    pub sigma_m_mpa: f64,
}

/// The extracted TLP bondline hotspot state: `Δε_t/2 = 0.001925`,
/// `Δε_p/2 = 0.000560`, `Δε_e/2 = 0.001365`, `σ_m = +42 MPa`.
pub const TLP_HOTSPOT: TlpHotspotStrain = TlpHotspotStrain {
    total_strain_amp: 0.001_925,
    plastic_strain_amp: 0.000_560,
    elastic_strain_amp: 0.001_365,
    sigma_m_mpa: 42.0,
};

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
        assert!((m.yield_strength(298.15) - 215.0).abs() < 1e-6);
        assert!((m.yield_strength(623.15) - 142.0).abs() < 1e-4);
        assert!((m.c1_modulus(298.15) - 45.2e3).abs() < 1e-6);
        assert!((m.c1_modulus(623.15) - 31.0e3).abs() < 0.05e3);
        assert!((m.c2_modulus(298.15) - 8.5e3).abs() < 1e-6);
        assert!((m.c2_modulus(623.15) - 5.2e3).abs() < 0.02e3);
        assert_eq!(m.back_stress(), 0.0);
    }

    #[test]
    fn stack_layer_table_matches_cf3() {
        let tlp = StackLayer::TlpBondline.at(298.15);
        assert_eq!(tlp.c1_gpa, 62.0);
        let cu = StackLayer::OfhcCu.at(623.15);
        assert!((cu.yield_mpa - 42.0).abs() < 1e-9);
    }

    #[test]
    fn vcct_delamination_margin_holds() {
        let f = vcct_delamination_factor(tlp_bondline_peak_release_rates());
        assert!((f - 0.4837).abs() < 1e-3);
        assert!(f < 1.0);
    }

    #[test]
    fn coffin_manson_morrow_life_bounds() {
        // Mean-stress-corrected solve of the cf3 strain-life equation.
        let nf = TLP_STRAIN_LIFE
            .cycles_to_failure(TLP_HOTSPOT.total_strain_amp, TLP_HOTSPOT.sigma_m_mpa);
        assert!(nf > 4.0e4 && nf < 1.0e5, "N_f = {nf}");
        // Zero mean stress recovers the ductility-dominated upper bound.
        let nf_nominal = TLP_STRAIN_LIFE.cycles_to_failure(TLP_HOTSPOT.total_strain_amp, 0.0);
        assert!(nf_nominal > FATIGUE_LIFE_REQUIRED);
    }

    #[test]
    fn initial_increment_bound_enforced() {
        ChabocheViscoplasticModel::assert_initial_increment(1.12e-3);
    }

    #[test]
    #[should_panic(expected = "single-cycle bound")]
    fn initial_increment_over_bound_panics() {
        ChabocheViscoplasticModel::assert_initial_increment(2.0e-3);
    }

    #[test]
    fn dual_backstress_evolves() {
        let mut m = ChabocheViscoplasticModel::pd_ir();
        m.evolve_backstress(1e-5, 1e-6, 0.0, 623.15, 1.0);
        assert!(m.back_stress_1 > 0.0 && m.back_stress_2 > 0.0);
        assert!(m.back_stress_1 > m.back_stress_2); // C1 > C2
    }
}
