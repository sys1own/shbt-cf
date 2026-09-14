//! Temperature-dependent material laws shared by the Belleville and thin-film
//! models: linear `E(T)` (cf.pdf Eq. 174) and linear `α(T)` (cf.pdf Eq. 215).

/// Absolute temperature of 0 °C in kelvin.
pub const KELVIN_OFFSET: f64 = 273.15;

/// Lower end of the qualified operating band (25 °C).
pub const T_COLD: f64 = 25.0 + KELVIN_OFFSET;
/// Upper end of the qualified operating band (350 °C).
pub const T_HOT: f64 = 350.0 + KELVIN_OFFSET;

/// Active core diameter [m].
pub const D_CORE: f64 = 0.020;
/// Active core contact area [m²].
pub const A_CONTACT: f64 = std::f64::consts::PI * D_CORE * D_CORE / 4.0;
/// Minimum clamp force for vacuum sealing [N].
pub const F_MIN: f64 = 6283.19;
/// Maximum clamp force before Pd-Ir film yield at 350 °C [N].
pub const F_MAX: f64 = 78539.82;
/// Optimized Inconel X-750 stack stiffness [N/m].
pub const K_STACK: f64 = 5.0e6;
/// Target net thermal mismatch deflection [m].
pub const DL_NET: f64 = 20.1e-6;
/// Plastic strain ceiling for the 44,820-cycle fatigue target.
pub const PLASTIC_STRAIN_CEILING: f64 = 0.0004805;

/// Structural result used by the thermomechanical integrity gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeaOutput {
    /// Maximum displacement [m].
    pub max_displacement: f64,
    /// Peak von Mises stress [Pa].
    pub peak_von_mises: f64,
    /// Cyclic plastic strain amplitude.
    pub cyclic_plastic_strain: f64,
    /// Calculated cycles to failure.
    pub calculated_cycles_to_failure: f64,
}

/// Validate the peak-temperature structural and fatigue limits.
pub fn validate_structural_design(output: &FeaOutput) -> Result<bool, &'static str> {
    if output.peak_von_mises > 250.0e6 {
        return Err("Interface stress exceeds yield strength of active Pd-Ir film at 350C");
    }
    if output.cyclic_plastic_strain > PLASTIC_STRAIN_CEILING {
        return Err("Low-cycle fatigue limit violated: expected lifespan below 44,820 cycles");
    }
    Ok(true)
}

/// Isotropic elastic constants at a fixed temperature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Elastic {
    /// Young's modulus [Pa].
    pub youngs: f64,
    /// Poisson's ratio.
    pub poisson: f64,
}

impl Elastic {
    /// Plane-stress biaxial modulus `E / (1 − ν)`.
    pub fn biaxial_modulus(self) -> f64 {
        self.youngs / (1.0 - self.poisson)
    }

    /// Plate modulus `E / (1 − ν²)`.
    pub fn plate_modulus(self) -> f64 {
        self.youngs / (1.0 - self.poisson * self.poisson)
    }
}

/// `E(T) = E₀ [1 − β (T − T_ref)]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearModulus {
    /// Modulus at `t_ref` [Pa].
    pub e0: f64,
    /// Fractional softening per kelvin.
    pub beta: f64,
    /// Reference temperature [K].
    pub t_ref: f64,
    /// Poisson's ratio (taken temperature independent).
    pub poisson: f64,
}

impl LinearModulus {
    /// Inconel X-750 spring alloy parameterisation of cf.pdf Eq. (174).
    pub const INCONEL_X750: Self = Self {
        e0: 193.0e9,
        beta: 2.8e-4,
        t_ref: 298.15,
        poisson: 0.30,
    };

    /// DIN 2093 spring steel reference (`E = 206 GPa`, `ν = 0.3`), used to
    /// generate the standard's force tables.
    pub const DIN_2093_STEEL: Self = Self {
        e0: 206.0e9,
        beta: 0.0,
        t_ref: 293.15,
        poisson: 0.30,
    };

    /// Elastic constants at absolute temperature `t` [K].
    pub fn at(self, t: f64) -> Elastic {
        Elastic {
            youngs: self.e0 * (1.0 - self.beta * (t - self.t_ref)),
            poisson: self.poisson,
        }
    }
}

/// Linear instantaneous coefficient of thermal expansion
/// `α(T) = a + b (T − T_ref)` under the reference-length engineering-strain
/// convention (cf.pdf Eq. 215).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearCte {
    /// CTE at `t_ref` [1/K].
    pub a: f64,
    /// CTE slope [1/K²].
    pub b: f64,
    /// Reference temperature [K].
    pub t_ref: f64,
}

impl LinearCte {
    /// Invar 36 reference (cf.pdf Eq. 216, `a_I`, `b_I`).
    pub const INVAR_36: Self = Self {
        a: 1.20e-6,
        b: 1.80e-9,
        t_ref: T_COLD,
    };
    /// Alloy reference (cf.pdf Eq. 216, `a_A`, `b_A`).
    pub const ALLOY_A: Self = Self {
        a: 2.35e-5,
        b: 1.50e-8,
        t_ref: T_COLD,
    };
    /// Precipitation-treated Inconel X-750 (cf.pdf Eq. 167, constant α).
    pub const INCONEL_X750: Self = Self {
        a: 14.2e-6,
        b: 0.0,
        t_ref: T_COLD,
    };
    /// CVD diamond (cf.pdf Table XXVII).
    pub const CVD_DIAMOND: Self = Self {
        a: 1.0e-6,
        b: 0.0,
        t_ref: T_COLD,
    };

    /// Instantaneous CTE at `t` [K].
    pub fn at(self, t: f64) -> f64 {
        self.a + self.b * (t - self.t_ref)
    }

    /// Exact integral `∫_{t0}^{t1} α(T) dT` (engineering thermal strain).
    pub fn strain(self, t0: f64, t1: f64) -> f64 {
        let d0 = t0 - self.t_ref;
        let d1 = t1 - self.t_ref;
        self.a * (d1 - d0) + 0.5 * self.b * (d1 * d1 - d0 * d0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inconel_modulus_matches_cf_eq_174() {
        let e = LinearModulus::INCONEL_X750.at(623.15).youngs;
        assert!((e - 175.437e9).abs() < 1e6, "{e}");
    }

    #[test]
    fn cte_integral_matches_cf_eq_216() {
        let d_i = 34.7021e-3 * LinearCte::INVAR_36.strain(T_COLD, T_HOT);
        let d_a = 1.0721e-3 * LinearCte::ALLOY_A.strain(T_COLD, T_HOT);
        assert!((d_i - 16.832_687_381e-6).abs() < 1e-14, "{d_i}");
        assert!((d_a - 9.037_467_969e-6).abs() < 1e-14, "{d_a}");
        assert!(((d_i - d_a) - 7.795_219_413e-6).abs() < 1e-14);
    }

    #[test]
    fn structural_design_rejects_yield_and_fatigue_violations() {
        let valid = FeaOutput {
            max_displacement: DL_NET,
            peak_von_mises: 250.0e6,
            cyclic_plastic_strain: PLASTIC_STRAIN_CEILING,
            calculated_cycles_to_failure: 44_820.0,
        };
        assert_eq!(validate_structural_design(&valid), Ok(true));

        let mut overstressed = valid;
        overstressed.peak_von_mises = 250.0e6 + 1.0;
        assert!(validate_structural_design(&overstressed).is_err());

        let mut fatigued = valid;
        fatigued.cyclic_plastic_strain = PLASTIC_STRAIN_CEILING + 1.0e-9;
        assert!(validate_structural_design(&fatigued).is_err());
    }
}
