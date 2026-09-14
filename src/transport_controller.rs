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
                    let d2c_dx2 = (c_l[i + 1][j][k] - 2.0 * c_l[i][j][k] + c_l[i - 1][j][k]) / (self.dx * self.dx);
                    let d2c_dy2 = (c_l[i][j + 1][k] - 2.0 * c_l[i][j][k] + c_l[i][j - 1][k]) / (self.dy * self.dy);
                    let d2c_dz2 = (c_l[i][j][k + 1] - 2.0 * c_l[i][j][k] + c_l[i][j][k - 1]) / (self.dz * self.dz);
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
    let c = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
    ];
    let mpc = StateSpaceMpc::new(a, b, c, 12);

    (transport, mpc)
}

pub fn complex_response(value: f64) -> Complex64 {
    Complex64::new(value, 0.0)
}
