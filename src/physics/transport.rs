#![allow(clippy::needless_range_loop)]

pub struct TransportParams {
    pub d_0: f64,       // m^2/s
    pub e_l: f64,       // J/mol
    pub v_h_star: f64,  // m^3/mol
    pub q_star: f64,    // J/mol
    pub k_s_0: f64,     // mol/(m^3 bar^0.5)
    pub delta_h_s: f64, // J/mol
    pub rho_m: f64,     // mol/m^3
    pub r_gas: f64,     // J/(mol K)
}

impl Default for TransportParams {
    fn default() -> Self {
        Self {
            d_0: 2.0e-7,
            e_l: 13500.0,
            v_h_star: 1.7e-6,
            q_star: -12000.0,
            k_s_0: 0.125,
            delta_h_s: 28600.0,
            rho_m: 1.34e5,
            r_gas: 8.314462618,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrapFamily {
    pub density: f64,        // mol/m^3
    pub binding_energy: f64, // J/mol
    pub k_c0: f64,           // m^3/(mol s)
    pub p_r0: f64,           // 1/s
}

#[derive(Clone, Debug)]
pub struct FvCellState {
    pub c_l: f64,                // mol/m^3
    pub c_t1: f64,               // mol/m^3
    pub c_t2: f64,               // mol/m^3
    pub temperature: f64,        // K
    pub hydrostatic_stress: f64, // Pa
}

#[derive(Clone, Debug)]
pub struct Mesh1D {
    pub centroids: Vec<f64>,
    pub volumes: Vec<f64>,
    pub dx: f64,
}

pub struct MicroTransportSolver {
    pub params: TransportParams,
    pub trap_1: TrapFamily,
    pub trap_2: TrapFamily,
    pub mesh: Mesh1D,
    pub state: Vec<FvCellState>,
    pub prev_state: Vec<FvCellState>,
    pub current_time: f64,
    pub dt: f64,
}

impl MicroTransportSolver {
    pub fn new(num_cells: usize, width: f64, initial_temp: f64) -> Self {
        let dx = width / (num_cells as f64);
        let mut centroids = Vec::with_capacity(num_cells);
        let mut volumes = Vec::with_capacity(num_cells);
        for i in 0..num_cells {
            centroids.push(((i as f64) + 0.5) * dx);
            volumes.push(dx);
        }

        let mesh = Mesh1D {
            centroids,
            volumes,
            dx,
        };
        let initial_state = vec![
            FvCellState {
                c_l: 1.0e-3,
                c_t1: 1.0e-4,
                c_t2: 1.0e-4,
                temperature: initial_temp,
                hydrostatic_stress: 0.0,
            };
            num_cells
        ];

        Self {
            params: TransportParams::default(),
            trap_1: TrapFamily {
                density: 50.0,
                binding_energy: 20000.0,
                k_c0: 1.0e-5,
                p_r0: 1.0e4,
            },
            trap_2: TrapFamily {
                density: 15.0,
                binding_energy: 55000.0,
                k_c0: 1.0e-5,
                p_r0: 0.0, // Irreversible trapping
            },
            mesh,
            state: initial_state.clone(),
            prev_state: initial_state,
            current_time: 0.0,
            dt: 1.0,
        }
    }

    pub fn compute_loading_ratio(&self, state: &FvCellState) -> f64 {
        (state.c_l + state.c_t1 + state.c_t2) / self.params.rho_m
    }

    pub fn compute_lattice_diffusivity(&self, temp: f64) -> f64 {
        self.params.d_0 * (-self.params.e_l / (self.params.r_gas * temp)).exp()
    }

    pub fn compute_sieverts_solubility(&self, temp: f64) -> f64 {
        self.params.k_s_0 * (-self.params.delta_h_s / (self.params.r_gas * temp)).exp()
    }

    pub fn enforce_sieverts_boundary(&mut self, p_d2_bar: f64, temp: f64) {
        let solubility = self.compute_sieverts_solubility(temp);
        let boundary_c_l = solubility * p_d2_bar.sqrt();
        if let Some(first) = self.state.first_mut() {
            first.c_l = boundary_c_l;
        }
        if let Some(last) = self.state.last_mut() {
            last.c_l = boundary_c_l;
        }
    }

    pub fn solve_step(&mut self) -> Result<(), String> {
        self.prev_state = self.state.clone();
        let num_cells = self.mesh.centroids.len();
        let max_iterations = 50;
        let tolerance = 1.0e-8;

        for iter in 0..max_iterations {
            let mut residual = vec![0.0; num_cells * 3];
            let mut jacobian = vec![vec![0.0; num_cells * 3]; num_cells * 3];

            for i in 0..num_cells {
                let cell_state = &self.state[i];
                let cell_prev = &self.prev_state[i];
                let temp = cell_state.temperature;
                let r_gas = self.params.r_gas;

                // Trap 1 Rate (Reversible)
                let k_c1 = self.trap_1.k_c0 * (-self.trap_1.binding_energy / (r_gas * temp)).exp();
                let p_r1 = self.trap_1.p_r0 * (-self.trap_1.binding_energy / (r_gas * temp)).exp();
                let trap_1_rate = k_c1 * cell_state.c_l * (self.trap_1.density - cell_state.c_t1)
                    - p_r1 * cell_state.c_t1;

                // Trap 2 Rate (Irreversible)
                let k_c2 = self.trap_2.k_c0 * (-self.trap_2.binding_energy / (r_gas * temp)).exp();
                let trap_2_rate = k_c2 * cell_state.c_l * (self.trap_2.density - cell_state.c_t2);

                let eq_l = i * 3;
                let eq_t1 = i * 3 + 1;
                let eq_t2 = i * 3 + 2;

                residual[eq_l] =
                    cell_state.c_l - cell_prev.c_l + self.dt * (trap_1_rate + trap_2_rate);
                residual[eq_t1] = cell_state.c_t1 - cell_prev.c_t1 - self.dt * trap_1_rate;
                residual[eq_t2] = cell_state.c_t2 - cell_prev.c_t2 - self.dt * trap_2_rate;

                jacobian[eq_l][eq_l] = 1.0
                    + self.dt
                        * (k_c1 * (self.trap_1.density - cell_state.c_t1)
                            + k_c2 * (self.trap_2.density - cell_state.c_t2));
                jacobian[eq_l][eq_t1] = -self.dt * (k_c1 * cell_state.c_l + p_r1);
                jacobian[eq_l][eq_t2] = -self.dt * k_c2 * cell_state.c_l;

                jacobian[eq_t1][eq_l] = -self.dt * k_c1 * (self.trap_1.density - cell_state.c_t1);
                jacobian[eq_t1][eq_t1] = 1.0 + self.dt * (k_c1 * cell_state.c_l + p_r1);

                jacobian[eq_t2][eq_l] = -self.dt * k_c2 * (self.trap_2.density - cell_state.c_t2);
                jacobian[eq_t2][eq_t2] = 1.0 + self.dt * k_c2 * cell_state.c_l;

                if i > 0 {
                    let flux = self.compute_intercell_flux(i - 1, i);
                    residual[eq_l] += (self.dt / self.mesh.volumes[i]) * flux;
                    jacobian[eq_l][eq_l] += (self.dt / self.mesh.volumes[i])
                        * (self.compute_flux_derivative_cl(i - 1, i, true));
                    jacobian[eq_l][(i - 1) * 3] += (self.dt / self.mesh.volumes[i])
                        * (self.compute_flux_derivative_cl(i - 1, i, false));
                }
                if i < num_cells - 1 {
                    let flux = self.compute_intercell_flux(i, i + 1);
                    residual[eq_l] -= (self.dt / self.mesh.volumes[i]) * flux;
                    jacobian[eq_l][eq_l] -= (self.dt / self.mesh.volumes[i])
                        * (self.compute_flux_derivative_cl(i, i + 1, false));
                    jacobian[eq_l][(i + 1) * 3] -= (self.dt / self.mesh.volumes[i])
                        * (self.compute_flux_derivative_cl(i, i + 1, true));
                }
            }

            let res_norm = residual.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if res_norm < tolerance {
                break;
            }

            if iter == max_iterations - 1 {
                return Err("Non-linear transport equations failed to converge".to_string());
            }

            let delta = self.solve_linear_system(&jacobian, &residual);
            for i in 0..num_cells {
                self.state[i].c_l -= delta[i * 3];
                self.state[i].c_t1 -= delta[i * 3 + 1];
                self.state[i].c_t2 -= delta[i * 3 + 2];

                self.state[i].c_l = self.state[i].c_l.max(0.0);
                self.state[i].c_t1 = self.state[i].c_t1.max(0.0).min(self.trap_1.density);
                self.state[i].c_t2 = self.state[i].c_t2.max(0.0).min(self.trap_2.density);
            }
        }

        for i in 0..num_cells {
            let x = self.compute_loading_ratio(&self.state[i]);
            if x > 0.904 {
                return Err(format!(
                    "Atomic loading ratio exceeds physical limit: x = {:.4}",
                    x
                ));
            }
        }

        self.current_time += self.dt;
        Ok(())
    }

    fn compute_intercell_flux(&self, idx_left: usize, idx_right: usize) -> f64 {
        let left = &self.state[idx_left];
        let right = &self.state[idx_right];

        let t_face = 0.5 * (left.temperature + right.temperature);
        let d_left = self.compute_lattice_diffusivity(left.temperature);
        let d_right = self.compute_lattice_diffusivity(right.temperature);
        let d_face = 0.5 * (d_left + d_right);

        let d_sigma = (right.hydrostatic_stress - left.hydrostatic_stress) / self.mesh.dx;
        let d_temp = (right.temperature - left.temperature) / self.mesh.dx;

        let v_drift = (d_face * self.params.v_h_star / (self.params.r_gas * t_face)) * d_sigma
            - (d_face * self.params.q_star / (self.params.r_gas * t_face * t_face)) * d_temp;

        let c_up = if v_drift >= 0.0 { left.c_l } else { right.c_l };
        let diffusive_flux = -d_face * (right.c_l - left.c_l) / self.mesh.dx;
        let drift_flux = v_drift * c_up;

        diffusive_flux + drift_flux
    }

    fn compute_flux_derivative_cl(
        &self,
        idx_left: usize,
        idx_right: usize,
        eval_right: bool,
    ) -> f64 {
        let left = &self.state[idx_left];
        let right = &self.state[idx_right];

        let t_face = 0.5 * (left.temperature + right.temperature);
        let d_left = self.compute_lattice_diffusivity(left.temperature);
        let d_right = self.compute_lattice_diffusivity(right.temperature);
        let d_face = 0.5 * (d_left + d_right);

        let d_sigma = (right.hydrostatic_stress - left.hydrostatic_stress) / self.mesh.dx;
        let d_temp = (right.temperature - left.temperature) / self.mesh.dx;

        let v_drift = (d_face * self.params.v_h_star / (self.params.r_gas * t_face)) * d_sigma
            - (d_face * self.params.q_star / (self.params.r_gas * t_face * t_face)) * d_temp;

        let mut deriv = if eval_right {
            -d_face / self.mesh.dx
        } else {
            d_face / self.mesh.dx
        };

        if v_drift >= 0.0 {
            if !eval_right {
                deriv += v_drift;
            }
        } else {
            if eval_right {
                deriv += v_drift;
            }
        }
        deriv
    }

    fn solve_linear_system(&self, a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let n = b.len();
        let mut mat = a.to_vec();
        let mut r = b.to_vec();

        for i in 0..n {
            let mut max_row = i;
            for k in i + 1..n {
                if mat[k][i].abs() > mat[max_row][i].abs() {
                    max_row = k;
                }
            }
            mat.swap(i, max_row);
            r.swap(i, max_row);

            for k in i + 1..n {
                let factor = mat[k][i] / mat[i][i];
                for j in i..n {
                    mat[k][j] -= factor * mat[i][j];
                }
                r[k] -= factor * r[i];
            }
        }

        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in i + 1..n {
                sum += mat[i][j] * x[j];
            }
            x[i] = (r[i] - sum) / mat[i][i];
        }
        x
    }

    pub fn step_with_temporal_rollback(&mut self) -> bool {
        let original_state = self.state.clone();
        let original_time = self.current_time;
        let original_dt = self.dt;

        match self.solve_step() {
            Ok(_) => true,
            Err(_) => {
                // Restore state and scale down timestep
                self.state = original_state;
                self.current_time = original_time;
                self.dt = original_dt * 0.5;
                false
            }
        }
    }
}
