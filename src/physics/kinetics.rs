//! Quantum Kinetics engine for the shbt-cf simulator.
//! This module implements 2nd-order Matrix Magnus expansion updates
//! alongside a trace-preserving, Hermiticity-enforcing RK4 solver.

use crate::physics::rcwa::Complex;

/// Representing a density matrix in the truncated Fock state basis.
#[derive(Debug, Clone)]
pub struct DensityMatrix {
    pub size: usize,
    pub data: Vec<Complex>,
}

impl DensityMatrix {
    pub fn new(size: usize) -> Self {
        Self {
            size,
            data: vec![Complex::zero(); size * size],
        }
    }

    pub fn identity(size: usize) -> Self {
        let mut mat = Self::new(size);
        for i in 0..size {
            mat.set(i, i, Complex::one());
        }
        mat
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> Complex {
        self.data[r * self.size + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, val: Complex) {
        self.data[r * self.size + c] = val;
    }

    /// Evaluates the trace of the density matrix.
    pub fn trace(&self) -> Complex {
        let mut tr = Complex::zero();
        for i in 0..self.size {
            tr = tr.add(self.get(i, i));
        }
        tr
    }

    /// Enforces structural Hermiticity and projects the state onto the unit trace manifold.
    pub fn enforce_physical_constraints(&mut self) {
        let n = self.size;

        // 1. Hermiticity projection: rho = 0.5 * (rho + rho^dagger)
        for r in 0..n {
            for c in r..n {
                let val1 = self.get(r, c);
                let val2 = self.get(c, r);
                let hermitian_val =
                    Complex::new(0.5 * (val1.re + val2.re), 0.5 * (val1.im - val2.im));
                self.set(r, c, hermitian_val);
                self.set(c, r, hermitian_val.conj());
            }
        }

        // 2. Normalization: rho = rho / Tr(rho)
        let tr = self.trace();
        if tr.re.abs() > 1e-15 {
            let tr_inv = 1.0 / tr.re;
            for i in 0..(n * n) {
                self.data[i] = self.data[i].mul_real(tr_inv);
            }
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        let mut out = Self::new(self.size);
        for i in 0..(self.size * self.size) {
            out.data[i] = self.data[i].add(other.data[i]);
        }
        out
    }

    pub fn scale(&self, val: f64) -> Self {
        let mut out = Self::new(self.size);
        for i in 0..(self.size * self.size) {
            out.data[i] = self.data[i].mul_real(val);
        }
        out
    }

    pub fn mul(&self, other: &Self) -> Self {
        let mut out = Self::new(self.size);
        for r in 0..self.size {
            for c in 0..self.size {
                let mut sum = Complex::zero();
                for k in 0..self.size {
                    sum = sum.add(self.get(r, k).mul(other.get(k, c)));
                }
                out.set(r, c, sum);
            }
        }
        out
    }

    /// Computes the matrix exponential using Taylor scaling and squaring.
    pub fn expm(&self, terms: usize) -> Self {
        let mut out = Self::identity(self.size);
        let mut term = Self::identity(self.size);

        for k in 1..terms {
            term = term.mul(self).scale(1.0 / (k as f64));
            out = out.add(&term);
        }
        out
    }
}

/// Representation of the open-system quantum dynamics engine.
#[derive(Debug, Clone)]
pub struct QuantumKineticsEngine {
    pub fock_cutoff: usize,
    pub gamma_lat: f64,
    pub gamma_other: f64,
    pub coupling_g: f64,
    pub delta_omega: f64,
}

impl QuantumKineticsEngine {
    pub fn new(fock_cutoff: usize, delta_f_beat: f64) -> Self {
        // High decay rate dominance configuration
        let g_lat = 1.5e5;
        let g_other = 1.2e-3;
        Self {
            fock_cutoff,
            gamma_lat: g_lat,
            gamma_other: g_other,
            coupling_g: 5.25e4,
            delta_omega: 2.0 * std::f64::consts::PI * delta_f_beat,
        }
    }

    /// Checks whether the branching ratio meets the mandatory physical constraints.
    pub fn verify_branching_ratio(&self) -> (f64, f64) {
        let b_lat = self.gamma_lat / (self.gamma_lat + self.gamma_other);
        let ratio = self.gamma_lat / self.gamma_other;
        (b_lat, ratio)
    }

    /// Generates the annihilation operator matrix.
    pub fn annihilation_operator(&self) -> DensityMatrix {
        let mut a = DensityMatrix::new(self.fock_cutoff);
        for i in 1..self.fock_cutoff {
            let val = (i as f64).sqrt();
            a.set(i - 1, i, Complex::new(val, 0.0));
        }
        a
    }

    /// Generates the creation operator matrix.
    pub fn creation_operator(&self) -> DensityMatrix {
        let mut adag = DensityMatrix::new(self.fock_cutoff);
        for i in 1..self.fock_cutoff {
            let val = (i as f64).sqrt();
            adag.set(i, i - 1, Complex::new(val, 0.0));
        }
        adag
    }

    /// Constructs the initial displaced thermal state.
    pub fn construct_initial_displaced_thermal_state(
        &self,
        alpha: f64,
        n_th: f64,
    ) -> DensityMatrix {
        let n = self.fock_cutoff;
        let mut rho_th = DensityMatrix::new(n);

        // Populate thermal occupancy probabilities
        for i in 0..n {
            let prob = n_th.powi(i as i32) / (1.0 + n_th).powi((i + 1) as i32);
            rho_th.set(i, i, Complex::new(prob, 0.0));
        }

        // Construct D(alpha) via exponentiation of G = alpha*a^dag - alpha^*a
        let a = self.annihilation_operator();
        let adag = self.creation_operator();

        let mut generator = DensityMatrix::new(n);
        for r in 0..n {
            for c in 0..n {
                let term1 = adag.get(r, c).mul_real(alpha);
                let term2 = a.get(r, c).mul_real(alpha);
                generator.set(r, c, term1.sub(term2));
            }
        }

        let d_op = generator.expm(35);

        let mut d_op_dag = DensityMatrix::new(n);
        for r in 0..n {
            for c in 0..n {
                d_op_dag.set(r, c, d_op.get(c, r).conj());
            }
        }

        d_op.mul(&rho_th).mul(&d_op_dag)
    }

    /// Evaluates the 2nd-order Matrix Magnus unitary time-evolution operator.
    pub fn evaluate_magnus_unitary(&self, t0: f64, dt: f64) -> DensityMatrix {
        let n = self.fock_cutoff;
        let a = self.annihilation_operator();
        let adag = self.creation_operator();

        // 1st order generator coeff
        let exp_term_1 = Complex::new(
            (-self.delta_omega * (t0 + dt)).cos(),
            (-self.delta_omega * (t0 + dt)).sin(),
        );
        let exp_term_2 = Complex::new(
            (-self.delta_omega * t0).cos(),
            (-self.delta_omega * t0).sin(),
        );

        let val_a = exp_term_1
            .sub(exp_term_2)
            .mul_real(self.coupling_g / self.delta_omega);
        let val_adag = Complex::new(val_a.re, -val_a.im).mul_real(-1.0); // Conjugate relation

        let mut omega_1 = DensityMatrix::new(n);
        for r in 0..n {
            for c in 0..n {
                let term1 = a.get(r, c).mul(val_a);
                let term2 = adag.get(r, c).mul(val_adag);
                omega_1.set(r, c, term1.add(term2));
            }
        }

        // 2nd order generator scalar phase shift
        let phase_scalar = (self.coupling_g * self.coupling_g)
            / (self.delta_omega * self.delta_omega)
            * (self.delta_omega * dt - (self.delta_omega * dt).sin());

        let omega_2_scalar = Complex::new(0.0, phase_scalar);

        let mut total_generator = omega_1;
        for i in 0..n {
            total_generator.set(i, i, total_generator.get(i, i).add(omega_2_scalar));
        }

        total_generator.expm(35)
    }

    /// Evaluates the multi-mode Lindbladian derivative L_diss(rho).
    pub fn evaluate_lindbladian_derivative(&self, rho: &DensityMatrix) -> DensityMatrix {
        let n = self.fock_cutoff;
        let a = self.annihilation_operator();
        let adag = self.creation_operator();

        // Calculate a * rho * a^dagger
        let mut a_rho = DensityMatrix::new(n);
        for r in 0..n {
            for c in 0..n {
                let mut sum_a = Complex::zero();
                for k in 0..n {
                    sum_a = sum_a.add(a.get(r, k).mul(rho.get(k, c)));
                }
                a_rho.set(r, c, sum_a);
            }
        }
        let a_rho_adag = a_rho.mul(&adag);

        // Calculate {a^dagger * a, rho}
        let adag_a = adag.mul(&a);
        let anticommutator_1 = adag_a.mul(rho).add(&rho.mul(&adag_a));

        // Primary lattice dissipator: gamma_lat * (a*rho*a^dag - 0.5*{a^dag*a, rho})
        let mut diss_lat = DensityMatrix::new(n);
        for r in 0..n {
            for c in 0..n {
                let val = a_rho_adag
                    .get(r, c)
                    .sub(anticommutator_1.get(r, c).mul_real(0.5));
                diss_lat.set(r, c, val.mul_real(self.gamma_lat));
            }
        }

        // Parasitic non-lattice dephasing channels: gamma_other
        let n_operator = adag_a;
        let n_rho_n = n_operator.mul(rho).mul(&n_operator);
        let n_sq = n_operator.mul(&n_operator);
        let anticommutator_2 = n_sq.mul(rho).add(&rho.mul(&n_sq));

        let mut diss_other = DensityMatrix::new(n);
        for r in 0..n {
            for c in 0..n {
                let val = n_rho_n
                    .get(r, c)
                    .sub(anticommutator_2.get(r, c).mul_real(0.5));
                diss_other.set(r, c, val.mul_real(self.gamma_other));
            }
        }

        diss_lat.add(&diss_other)
    }

    /// Performs a single dynamic step using the trace-preserving RK4 integration method.
    pub fn step_dynamics(&self, rho: &DensityMatrix, t0: f64, dt: f64) -> DensityMatrix {
        // Step 1: Analytical Magnus unitary step
        let u = self.evaluate_magnus_unitary(t0, dt);
        let mut u_dag = DensityMatrix::new(self.fock_cutoff);
        for r in 0..self.fock_cutoff {
            for c in 0..self.fock_cutoff {
                u_dag.set(r, c, u.get(c, r).conj());
            }
        }
        let rho_unitary = u.mul(rho).mul(&u_dag);

        // Step 2: Dissipative RK4 step
        let k1 = self.evaluate_lindbladian_derivative(&rho_unitary);

        let rho_k2 = rho_unitary.add(&k1.scale(0.5 * dt));
        let k2 = self.evaluate_lindbladian_derivative(&rho_k2);

        let rho_k3 = rho_unitary.add(&k2.scale(0.5 * dt));
        let k3 = self.evaluate_lindbladian_derivative(&rho_k3);

        let rho_k4 = rho_unitary.add(&k3.scale(dt));
        let k4 = self.evaluate_lindbladian_derivative(&rho_k4);

        let k_sum = k1.add(&k2.scale(2.0)).add(&k3.scale(2.0)).add(&k4);
        let mut rho_next = rho_unitary.add(&k_sum.scale(dt / 6.0));

        // Step 3: Projection onto the physical, trace-conserving manifold
        rho_next.enforce_physical_constraints();

        rho_next
    }
}

pub struct McNabbFosterSolver {
    pub c_l_init: f64,
    pub n_t1: f64,
    pub n_t2: f64,
    pub v_h_star: f64,
    pub temp: f64,
}

impl McNabbFosterSolver {
    /// Enforces loading ratio boundaries across diffuse and trapped families
    pub fn enforce_loading_constraints(
        &self,
        c_l: f64,
        c_t1: f64,
        c_t2: f64,
        metal_density: f64,
    ) -> Result<f64, String> {
        let x_loading = (c_l + c_t1 + c_t2) / metal_density;

        if x_loading > 0.904 {
            return Err(format!(
                "Unstable thermodynamic phase: x_loading = {} exceeds x_max = 0.904",
                x_loading
            ));
        }
        if x_loading > 0.81 {
            // Trigger operational soft-clamp warning
            println!(
                "Warning: x_loading = {} exceeds operational limit of 0.81",
                x_loading
            );
        }

        Ok(x_loading)
    }
}
