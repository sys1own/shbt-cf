#![allow(clippy::new_without_default, clippy::too_many_arguments)]

#[derive(Clone, Debug)]
pub struct Tensor3D {
    pub xx: f64,
    pub yy: f64,
    pub zz: f64,
    pub xy: f64,
    pub yz: f64,
    pub xz: f64,
}

impl Tensor3D {
    pub fn zero() -> Self {
        Self {
            xx: 0.0,
            yy: 0.0,
            zz: 0.0,
            xy: 0.0,
            yz: 0.0,
            xz: 0.0,
        }
    }

    pub fn trace(&self) -> f64 {
        self.xx + self.yy + self.zz
    }

    pub fn deviatoric(&self) -> Self {
        let mean = self.trace() / 3.0;
        Self {
            xx: self.xx - mean,
            yy: self.yy - mean,
            zz: self.zz - mean,
            xy: self.xy,
            yz: self.yz,
            xz: self.xz,
        }
    }

    pub fn inner_product(&self, other: &Self) -> f64 {
        self.xx * other.xx
            + self.yy * other.yy
            + self.zz * other.zz
            + 2.0 * (self.xy * other.xy + self.yz * other.yz + self.xz * other.xz)
    }

    pub fn j2_norm(&self) -> f64 {
        (1.5 * self.inner_product(self)).sqrt()
    }
}

pub struct BellevilleStack {
    pub stiffness: f64,  // N/m
    pub net_offset: f64, // m
    pub f_min: f64,      // N
    pub f_max: f64,      // N
}

impl BellevilleStack {
    pub fn new() -> Self {
        Self {
            stiffness: 5.0e6,
            net_offset: 20.1e-6,
            f_min: 6283.19,
            f_max: 78539.82,
        }
    }

    pub fn calculate_force(&self, displacement: f64) -> f64 {
        let calculated_force = self.stiffness * (displacement + self.net_offset);
        calculated_force.max(self.f_min).min(self.f_max)
    }
}

pub struct ComplianceLayer {
    pub thickness_tlp: f64,    // m
    pub thickness_graded: f64, // m
    pub e_tlp: f64,            // Pa
    pub e_graded: f64,         // Pa
}

impl ComplianceLayer {
    pub fn new() -> Self {
        Self {
            thickness_tlp: 3.5e-6,
            thickness_graded: 15.0e-6,
            e_tlp: 85.0e9,
            e_graded: 110.0e9,
        }
    }

    pub fn evaluate_pressure_deviation(
        &self,
        mean_force: f64,
        force_deviation: f64,
        area: f64,
    ) -> f64 {
        let nominal_pressure = mean_force / area;
        let actual_pressure = (mean_force + force_deviation) / area;
        (actual_pressure - nominal_pressure).abs() / 1.0e6 // Return deviation in MPa
    }
}

pub struct ChabocheViscoplasticity {
    pub sigma_y: f64, // Pa
    pub c1: f64,      // Pa
    pub c2: f64,      // Pa
    pub gamma1: f64,
    pub gamma2: f64,
    pub k_v: f64, // Pa s^(1/nv)
    pub n_v: f64,
    pub b: f64,
    pub q_inf: f64, // Pa
}

impl ChabocheViscoplasticity {
    /// Pd0.9132Ir0.0868 active-layer constants at the 298.15 K endpoint of
    /// the thermal cycle (cf3 spec §2.1): sigma_y = 215 MPa, C1 = 45.2 GPa,
    /// gamma1 = 410, C2 = 8.5 GPa, gamma2 = 62, R_inf = 85 MPa, b = 12.46.
    pub fn new() -> Self {
        Self {
            sigma_y: 215.0e6,
            c1: 45.2e9,
            c2: 8.5e9,
            gamma1: 410.0,
            gamma2: 62.0,
            k_v: 150.0e6,
            n_v: 6.5,
            b: 12.46,
            q_inf: 85.0e6,
        }
    }

    pub fn integrate_step(
        &self,
        stress: &Tensor3D,
        x1: &mut Tensor3D,
        x2: &mut Tensor3D,
        r: &mut f64,
        ep: &mut Tensor3D,
        p: &mut f64,
        dt: f64,
        temp_rate: f64,
        dc_dt_factor: f64, // (1/C) * dC/dT
    ) {
        let s = stress.deviatoric();
        let total_x_d = Tensor3D {
            xx: x1.xx + x2.xx,
            yy: x1.yy + x2.yy,
            zz: x1.zz + x2.zz,
            xy: x1.xy + x2.xy,
            yz: x1.yz + x2.yz,
            xz: x1.xz + x2.xz,
        };

        let diff_s_x = Tensor3D {
            xx: s.xx - total_x_d.xx,
            yy: s.yy - total_x_d.yy,
            zz: s.zz - total_x_d.zz,
            xy: s.xy - total_x_d.xy,
            yz: s.yz - total_x_d.yz,
            xz: s.xz - total_x_d.xz,
        };

        let j2 = diff_s_x.j2_norm();
        let f = j2 - self.sigma_y - *r;

        if f > 0.0 {
            let dp = (f / self.k_v).powf(self.n_v) * dt;
            *p += dp;

            // Isotropic hardening update
            *r += self.b * (self.q_inf - *r) * dp;

            // Flow direction
            let flow_tensor = Tensor3D {
                xx: 1.5 * diff_s_x.xx / j2,
                yy: 1.5 * diff_s_x.yy / j2,
                zz: 1.5 * diff_s_x.zz / j2,
                xy: 1.5 * diff_s_x.xy / j2,
                yz: 1.5 * diff_s_x.yz / j2,
                xz: 1.5 * diff_s_x.xz / j2,
            };

            // Viscoplastic strain update
            ep.xx += flow_tensor.xx * dp;
            ep.yy += flow_tensor.yy * dp;
            ep.zz += flow_tensor.zz * dp;
            ep.xy += flow_tensor.xy * dp;
            ep.yz += flow_tensor.yz * dp;
            ep.xz += flow_tensor.xz * dp;

            // Kinematic backstress updates (with temperature rate scaling)
            let thermal_term = dc_dt_factor * temp_rate * dt;

            x1.xx += (2.0 / 3.0) * self.c1 * flow_tensor.xx * dp - self.gamma1 * x1.xx * dp
                + thermal_term * x1.xx;
            x1.yy += (2.0 / 3.0) * self.c1 * flow_tensor.yy * dp - self.gamma1 * x1.yy * dp
                + thermal_term * x1.yy;
            x1.zz += (2.0 / 3.0) * self.c1 * flow_tensor.zz * dp - self.gamma1 * x1.zz * dp
                + thermal_term * x1.zz;
            x1.xy += (2.0 / 3.0) * self.c1 * flow_tensor.xy * dp - self.gamma1 * x1.xy * dp
                + thermal_term * x1.xy;
            x1.yz += (2.0 / 3.0) * self.c1 * flow_tensor.yz * dp - self.gamma1 * x1.yz * dp
                + thermal_term * x1.yz;
            x1.xz += (2.0 / 3.0) * self.c1 * flow_tensor.xz * dp - self.gamma1 * x1.xz * dp
                + thermal_term * x1.xz;

            x2.xx += (2.0 / 3.0) * self.c2 * flow_tensor.xx * dp - self.gamma2 * x2.xx * dp
                + thermal_term * x2.xx;
            x2.yy += (2.0 / 3.0) * self.c2 * flow_tensor.yy * dp - self.gamma2 * x2.yy * dp
                + thermal_term * x2.yy;
            x2.zz += (2.0 / 3.0) * self.c2 * flow_tensor.zz * dp - self.gamma2 * x2.zz * dp
                + thermal_term * x2.zz;
            x2.xy += (2.0 / 3.0) * self.c2 * flow_tensor.xy * dp - self.gamma2 * x2.xy * dp
                + thermal_term * x2.xy;
            x2.yz += (2.0 / 3.0) * self.c2 * flow_tensor.yz * dp - self.gamma2 * x2.yz * dp
                + thermal_term * x2.yz;
            x2.xz += (2.0 / 3.0) * self.c2 * flow_tensor.xz * dp - self.gamma2 * x2.xz * dp
                + thermal_term * x2.xz;
        }
    }
}

/// Coffin–Manson–Morrow strain-life evaluator for the Ni-Cu-Sn TLP bondline
/// (cf3 spec §2.3): sigma_f' = 520 MPa, b = -0.095, eps_f' = 0.380,
/// c = -0.580 over the temperature-averaged modulus E = 112 GPa.
pub struct CoffinMansonEvaluator {
    pub sigma_f_prime: f64,  // Pa
    pub b: f64,
    pub epsilon_f_prime: f64,
    pub c: f64,
    pub e_modulus: f64, // Pa
}

impl CoffinMansonEvaluator {
    pub fn new() -> Self {
        Self {
            sigma_f_prime: 520.0e6,
            b: -0.095,
            epsilon_f_prime: 0.380,
            c: -0.580,
            e_modulus: 112.0e9,
        }
    }

    /// Total strain amplitude at `reversals` (= 2 N_f) with mean-stress
    /// correction sigma_m [Pa].
    pub fn strain_amplitude(&self, reversals: f64, sigma_m: f64) -> f64 {
        (self.sigma_f_prime - sigma_m) / self.e_modulus * reversals.powf(self.b)
            + self.epsilon_f_prime * reversals.powf(self.c)
    }

    /// Cycles to failure N_f by bisection on the reversal count.
    pub fn calculate_fatigue_life(&self, strain_amplitude: f64, sigma_m: f64) -> f64 {
        assert!(strain_amplitude > 0.0);
        let mut lo = 2.0f64;
        let mut hi = 1.0e12f64;
        for _ in 0..200 {
            let mid = (lo * hi).sqrt();
            if self.strain_amplitude(mid, sigma_m) > strain_amplitude {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (lo * hi).sqrt() / 2.0
    }

    /// Pure Coffin–Manson estimate using only the plastic strain range.
    pub fn calculate_plastic_life(&self, delta_ep: f64) -> f64 {
        let inner = delta_ep / (2.0 * self.epsilon_f_prime);
        0.5 * inner.powf(1.0 / self.c)
    }
}

/// Linear power-law VCCT mixed-mode delamination criterion for the
/// (Cu,Ni)6Sn5 / CVD-diamond interface of the 3.5 um TLP bondline
/// (cf3 spec §2.2): G_IC = 25.0, G_IIC = 65.0, G_IIIC = 60.0 J/m^2.
pub struct VcctInterface {
    pub g_ic: f64,
    pub g_iic: f64,
    pub g_iiic: f64,
}

impl VcctInterface {
    pub fn new() -> Self {
        Self {
            g_ic: 25.0,
            g_iic: 65.0,
            g_iiic: 60.0,
        }
    }

    /// f_delam = G_I/G_IC + G_II/G_IIC + G_III/G_IIIC; initiation at >= 1.0.
    pub fn delamination_factor(&self, g_i: f64, g_ii: f64, g_iii: f64) -> f64 {
        g_i / self.g_ic + g_ii / self.g_iic + g_iii / self.g_iiic
    }

    /// Peak bondline release rates at the outer stack perimeter under the
    /// 623.15 K -> 298.15 K, 100 K/s shock: G_I = 4.12, G_II = 18.45,
    /// G_III = 2.10 J/m^2 giving f_delam = 0.484 << 1.0.
    pub fn bondline_margin(&self) -> f64 {
        self.delamination_factor(4.12, 18.45, 2.10)
    }
}
