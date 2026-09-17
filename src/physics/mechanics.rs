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
    pub fn new() -> Self {
        Self {
            sigma_y: 120.0e6,
            c1: 45.0e9,
            c2: 8.5e9,
            gamma1: 500.0,
            gamma2: 45.0,
            k_v: 150.0e6,
            n_v: 6.5,
            b: 4.2,
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

pub struct CoffinMansonEvaluator {
    pub epsilon_f_prime: f64,
    pub c: f64,
}

impl CoffinMansonEvaluator {
    pub fn new() -> Self {
        Self {
            epsilon_f_prime: 0.35,
            c: -0.51,
        }
    }

    pub fn calculate_fatigue_life(&self, delta_ep: f64) -> f64 {
        let inner = delta_ep / (2.0 * self.epsilon_f_prime);
        0.5 * inner.powf(1.0 / self.c)
    }
}
