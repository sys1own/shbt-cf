use crate::types::Complex64;

pub struct FloquetMatrixInverter {
    pub num_g: usize,
    pub num_floquet: usize,
    pub omega_d: f64,
    pub e_field_amp: f64,
    pub m_eff: f64,
}

impl FloquetMatrixInverter {
    pub fn new(
        num_g: usize,
        num_floquet: usize,
        omega_d: f64,
        e_field_amp: f64,
        m_eff: f64,
    ) -> Self {
        Self {
            num_g,
            num_floquet,
            omega_d,
            e_field_amp,
            m_eff,
        }
    }

    pub fn total_dim(&self) -> usize {
        self.num_g * self.num_floquet
    }

    /// Assembles the full composite reciprocal-space / Floquet dielectric matrix.
    pub fn assemble_dielectric_matrix(
        &self,
        q_vec: [f64; 3],
        g_vectors: &[[f64; 3]],
    ) -> Vec<Vec<Complex64>> {
        let dim = self.total_dim();
        let mut mat = vec![vec![Complex64::zero(); dim]; dim];

        let e_charge = 1.602176634e-19;
        let eps_0 = 8.8541878128e-12;
        let m_max = (self.num_floquet - 1) / 2;

        for g_i in 0..self.num_g {
            for m_i in 0..self.num_floquet {
                let row = g_i * self.num_floquet + m_i;
                let m_val = m_i as isize - m_max as isize;

                let qg = [
                    q_vec[0] + g_vectors[g_i][0],
                    q_vec[1] + g_vectors[g_i][1],
                    q_vec[2] + g_vectors[g_i][2],
                ];
                let qg_sq = qg[0] * qg[0] + qg[1] * qg[1] + qg[2] * qg[2];

                for g_j in 0..self.num_g {
                    for n_j in 0..self.num_floquet {
                        let col = g_j * self.num_floquet + n_j;
                        let n_val = n_j as isize - m_max as isize;

                        if row == col {
                            mat[row][col] = Complex64::one();
                        } else if g_i == g_j {
                            let v_c = e_charge * e_charge / (eps_0 * qg_sq.max(1e-12));
                            let chi_val = 0.05 / (1.0 + (m_val - n_val).abs() as f64);
                            mat[row][col] = mat[row][col].sub(Complex64::new(v_c * chi_val, 0.0));
                        }
                    }
                }
            }
        }
        mat
    }

    /// Solves a complex linear system via Gauss-Jordan elimination with partial pivoting.
    pub fn invert_matrix(&self, mat: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
        let n = mat.len();
        let mut aug = vec![vec![Complex64::zero(); 2 * n]; n];

        for i in 0..n {
            for j in 0..n {
                aug[i][j] = mat[i][j];
            }
            aug[i][n + i] = Complex64::one();
        }

        for i in 0..n {
            let mut pivot = i;
            let mut max_norm = aug[i][i].re * aug[i][i].re + aug[i][i].im * aug[i][i].im;
            for k in (i + 1)..n {
                let norm = aug[k][i].re * aug[k][i].re + aug[k][i].im * aug[k][i].im;
                if norm > max_norm {
                    max_norm = norm;
                    pivot = k;
                }
            }

            if max_norm < 1e-28 {
                panic!("Dielectric matrix is singular or ill-conditioned");
            }

            aug.swap(i, pivot);

            let pivot_val = aug[i][i];
            for j in 0..(2 * n) {
                aug[i][j] = aug[i][j].div(pivot_val);
            }

            for k in 0..n {
                if k != i {
                    let factor = aug[k][i];
                    for j in 0..(2 * n) {
                        let sub_val = factor.mul(aug[i][j]);
                        aug[k][j] = aug[k][j].sub(sub_val);
                    }
                }
            }
        }

        let mut inv = vec![vec![Complex64::zero(); n]; n];
        for i in 0..n {
            for j in 0..n {
                inv[i][j] = aug[i][n + j];
            }
        }
        inv
    }

    /// Extracts the static elastic block and computes the effective screening shift in eV.
    pub fn compute_effective_screening(
        &self,
        r_d: f64,
        q_vec: [f64; 3],
        g_vectors: &[[f64; 3]],
    ) -> f64 {
        let mat = self.assemble_dielectric_matrix(q_vec, g_vectors);
        let inv = self.invert_matrix(&mat);

        let m_max = (self.num_floquet - 1) / 2;
        let zero_sideband_idx = m_max;

        let mut re_w_00 = 0.0;
        let e_charge = 1.602176634e-19;
        let eps_0 = 8.8541878128e-12;

        for g_i in 0..self.num_g {
            let row = g_i * self.num_floquet + zero_sideband_idx;
            for g_j in 0..self.num_g {
                let col = g_j * self.num_floquet + zero_sideband_idx;
                let inv_elem = inv[row][col];

                let qg = [
                    q_vec[0] + g_vectors[g_j][0],
                    q_vec[1] + g_vectors[g_j][1],
                    q_vec[2] + g_vectors[g_j][2],
                ];
                let qg_sq = qg[0] * qg[0] + qg[1] * qg[1] + qg[2] * qg[2];
                let v_c = e_charge * e_charge / (eps_0 * qg_sq.max(1e-12));

                re_w_00 += inv_elem.re * v_c;
            }
        }

        let v_bare_joules = e_charge * e_charge / (4.0 * std::f64::consts::PI * eps_0 * r_d);
        let v_bare_ev = v_bare_joules / e_charge;
        let v_driven_ev = re_w_00 / e_charge;

        v_bare_ev - v_driven_ev
    }
}
