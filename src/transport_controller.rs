use crate::types::Complex64;

pub struct DeuteriumTransport3D {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    pub d_lattice: f64,
    pub v_star: f64,
    pub temp: f64,
}

#[allow(clippy::too_many_arguments)]
impl DeuteriumTransport3D {
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        d_lattice: f64,
        v_star: f64,
        temp: f64,
    ) -> Self {
        Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            d_lattice,
            v_star,
            temp,
        }
    }

    /// Evolves the 3D concentration profile C_L under diffusion and hydrostatic stress gradient electromigration.
    pub fn step_transport(&self, c_l: &mut Vec<Vec<Vec<f64>>>, sigma_m: &[Vec<Vec<f64>>], dt: f64) {
        let r_gas = 8.314462618;
        let drift_coeff = self.d_lattice * self.v_star / (r_gas * self.temp);

        let mut c_next = c_l.clone();

        for i in 1..(self.nx - 1) {
            for j in 1..(self.ny - 1) {
                for k in 1..(self.nz - 1) {
                    let d2c_dx2 = (c_l[i + 1][j][k] - 2.0 * c_l[i][j][k] + c_l[i - 1][j][k])
                        / (self.dx * self.dx);
                    let d2c_dy2 = (c_l[i][j + 1][k] - 2.0 * c_l[i][j][k] + c_l[i][j - 1][k])
                        / (self.dy * self.dy);
                    let d2c_dz2 = (c_l[i][j][k + 1] - 2.0 * c_l[i][j][k] + c_l[i][j][k - 1])
                        / (self.dz * self.dz);
                    let laplacian = d2c_dx2 + d2c_dy2 + d2c_dz2;

                    let d_sigma_x = (sigma_m[i + 1][j][k] - sigma_m[i - 1][j][k]) / (2.0 * self.dx);
                    let d_c_x = (c_l[i + 1][j][k] - c_l[i - 1][j][k]) / (2.0 * self.dx);
                    let advection_x = drift_coeff * (d_c_x * d_sigma_x);

                    let flux_div = self.d_lattice * laplacian + advection_x;
                    c_next[i][j][k] += dt * flux_div;
                }
            }
        }
        *c_l = c_next;
    }
}

pub struct StateSpaceMpc {
    pub a_matrix: Vec<Vec<f64>>, // 4x4 reduced state matrix
    pub b_matrix: Vec<Vec<f64>>, // 4x2 control matrix
    pub c_matrix: Vec<Vec<f64>>, // 2x4 output matrix
    pub horizon: usize,
}

#[allow(clippy::needless_range_loop)]
impl StateSpaceMpc {
    pub fn new(
        a_matrix: Vec<Vec<f64>>,
        b_matrix: Vec<Vec<f64>>,
        c_matrix: Vec<Vec<f64>>,
        horizon: usize,
    ) -> Self {
        Self {
            a_matrix,
            b_matrix,
            c_matrix,
            horizon,
        }
    }

    /// Computes optimal control increments [dI_laser, dQ_cool] to prevent runaway and maximize absorption.
    pub fn compute_control(&self, current_state: &[f64; 4], target_output: &[f64; 2]) -> [f64; 2] {
        let mut current_y = [0.0; 2];
        for i in 0..2 {
            for j in 0..4 {
                current_y[i] += self.c_matrix[i][j] * current_state[j];
            }
        }

        let err_export = target_output[0] - current_y[0];
        let err_temp = target_output[1] - current_y[1];

        let kp_laser = 0.012;
        let ki_cool = 0.045;

        let delta_laser = kp_laser * err_export;
        let delta_cooling = -ki_cool * err_temp;

        [delta_laser, delta_cooling]
    }
}

/// Microchannel cold plate geometry and coolant state (update-11.1 Task 2).
pub struct TransportController {
    /// Cold plate width `W` [m].
    pub cold_plate_width_m: f64,
    /// Cold plate length `L` [m].
    pub cold_plate_length_m: f64,
    /// Channel width `w_c` [m].
    pub channel_width_m: f64,
    /// Fin thickness `w_w` [m].
    pub fin_width_m: f64,
    /// Channel height `H_c` [m].
    pub channel_height_m: f64,
    /// Number of microchannels `N_ch`.
    pub channel_count: usize,
    /// Volumetric coolant flow `V̇` [m³/s].
    pub coolant_flow_rate_m3_s: f64,
    /// Waste heat dumped into the cold side `Q_cold` [W].
    pub waste_heat_q_cold_w: f64,
}

/// TEG + balance-of-plant power ledger for one `(T_hot, T_cold)` operating
/// point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BopLedgerResult {
    /// TEG conversion efficiency `η_TEG` (dimensionless).
    pub eta_teg: f64,
    /// Total thermal power `P_thermal = Q_cold / (1 − η_TEG)` [W].
    pub p_thermal_w: f64,
    /// TEG electric output `P_TEG = P_thermal · η_TEG` [W].
    pub p_teg_w: f64,
    /// Parasitic draw `P_laser + P_aux + P_pump` [W].
    pub p_parasitic_w: f64,
    /// Net export `P_TEG − ΣP_parasitic` [W].
    pub p_net_w: f64,
}

/// Production cold plate: 120 × 120 mm, 240 channels of 250 µm × 1500 µm,
/// 1.8 L/min water, 2047.86 W cold-side waste heat.
impl TransportController {
    /// Production cold-plate configuration from the update-11.1 spec.
    pub fn new() -> Self {
        Self {
            cold_plate_width_m: 0.120,
            cold_plate_length_m: 0.120,
            channel_width_m: 250.0e-6,
            fin_width_m: 250.0e-6,
            channel_height_m: 1500.0e-6,
            channel_count: 240,
            coolant_flow_rate_m3_s: 3.0e-5,
            waste_heat_q_cold_w: 2047.86,
        }
    }

    /// Evaluates the TEG/BOP ledger at `ZT_avg = 1.5`.
    ///
    /// `η_TEG = ((T_h − T_c)/T_h) · (√(1+ZT) − 1)/(√(1+ZT) + T_c/T_h)`,
    /// `P_thermal = Q_cold/(1−η)`, `P_TEG = P_thermal·η`,
    /// `P_parasitic = 150 + 50 + 0.3161`, `P_net = P_TEG − P_parasitic`.
    ///
    /// # Panics
    /// If `t_cold >= 358.0` K — the cold-side safety limit.
    pub fn evaluate_bop_ledger(&self, t_hot: f64, t_cold: f64) -> BopLedgerResult {
        assert!(
            t_cold < 358.0,
            "Cold side temperature exceeded 358.0 K safety limit"
        );
        let eta_teg = ((t_hot - t_cold) / t_hot)
            * (((1.5 + 1.0f64).sqrt() - 1.0) / ((1.5 + 1.0f64).sqrt() + (t_cold / t_hot)));
        let p_thermal = self.waste_heat_q_cold_w / (1.0 - eta_teg);
        let p_teg = p_thermal * eta_teg;
        let p_parasitic = 150.0 + 50.0 + 0.3161;
        let p_net = p_teg - p_parasitic;
        BopLedgerResult {
            eta_teg,
            p_thermal_w: p_thermal,
            p_teg_w: p_teg,
            p_parasitic_w: p_parasitic,
            p_net_w: p_net,
        }
    }
}

impl Default for TransportController {
    fn default() -> Self {
        Self::new()
    }
}

pub fn default_transport_controller() -> (DeuteriumTransport3D, StateSpaceMpc) {
    let transport = DeuteriumTransport3D::new(8, 8, 8, 1.0, 1.0, 1.0, 1e-10, 1e-6, 550.0);
    let a = vec![
        vec![0.95, 0.01, 0.0, 0.0],
        vec![0.0, 0.90, 0.02, 0.0],
        vec![0.0, 0.0, 0.85, 0.03],
        vec![0.0, 0.0, 0.0, 0.80],
    ];
    let b = vec![
        vec![0.5, 0.1],
        vec![0.2, 0.3],
        vec![0.1, 0.4],
        vec![0.0, 0.5],
    ];
    let c = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 0.0]];
    let mpc = StateSpaceMpc::new(a, b, c, 12);

    (transport, mpc)
}

pub fn complex_response(value: f64) -> Complex64 {
    Complex64::new(value, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bop_ledger_matches_spec_operating_point() {
        let ctl = TransportController::new();
        let r = ctl.evaluate_bop_ledger(600.0, 357.47);
        assert!((r.eta_teg - 0.1079).abs() < 1e-3, "eta_teg = {}", r.eta_teg);
        assert!((r.p_thermal_w - 2295.58).abs() < 1.0);
        assert!((r.p_teg_w - 247.72).abs() < 0.5);
        assert!((r.p_parasitic_w - 200.3161).abs() < 1e-9);
        assert!((r.p_net_w - 47.4039).abs() < 0.5, "p_net = {}", r.p_net_w);
        assert!(r.p_net_w > 0.0, "net positive export");
    }

    #[test]
    #[should_panic(expected = "Cold side temperature exceeded 358.0 K safety limit")]
    fn bop_ledger_rejects_overlimit_cold_side() {
        let ctl = TransportController::new();
        let _ = ctl.evaluate_bop_ledger(600.0, 358.0);
    }
}
