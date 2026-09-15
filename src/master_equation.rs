//! Three-state Lindblad master-equation solver coupled to the balance-of-plant
//! (BOP) thermal-hydraulic reconciliation (update-10.1 §2–§4).
//!
//! Basis: `|0>` ground, `|1>` phonon-dressed excited, `|2>` coherent
//! nuclear-coupled. The density matrix is a row-major `[Complex64; 9]` and
//! `ħ = 1` throughout.

#![allow(unsafe_code)]

use crate::types::Complex64;

/// Row-major 3×3 complex density matrix.
pub type Rho3 = [Complex64; 9];

/// Hamiltonian and dissipator parameters of the driven 3-state system.
///
/// ```text
/// H(t) = [ 0                    Ω₀ cos(ω_L t)   0       ]
///        [ Ω₀ cos(ω_L t)        Δ               g_nuc   ]
///        [ 0                    g_nuc           δ_nuc   ]
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct LindbladParams {
    /// Drive Rabi amplitude Ω₀.
    pub omega_0: f64,
    /// Drive angular frequency ω_L.
    pub omega_l: f64,
    /// Detuning Δ of `|1>`.
    pub delta: f64,
    /// Nuclear-lattice coupling g_nuc between `|1>` and `|2>`.
    pub g_nuc: f64,
    /// Detuning δ_nuc of `|2>`.
    pub det_nuc: f64,
    /// Rate of jump operator `L₁ = |0><1|`.
    pub gamma_1: f64,
    /// Rate of jump operator `L₂ = |1><2|`.
    pub gamma_2: f64,
}

#[inline]
fn scale(z: Complex64, s: f64) -> Complex64 {
    Complex64::new(z.re * s, z.im * s)
}

/// Ground-state density matrix `|0><0|`.
pub fn ground_state() -> Rho3 {
    let mut rho = [Complex64::zero(); 9];
    rho[0] = Complex64::one();
    rho
}

/// Right-hand side of `dρ/dt = -i[H(t), ρ] + γ₁ D[L₁](ρ) + γ₂ D[L₂](ρ)`.
pub fn compute_lindblad_derivative(rho: &Rho3, t: f64, p: &LindbladParams) -> Rho3 {
    let drive = p.omega_0 * (p.omega_l * t).cos();
    let h = [
        Complex64::zero(),
        Complex64::new(drive, 0.0),
        Complex64::zero(),
        Complex64::new(drive, 0.0),
        Complex64::new(p.delta, 0.0),
        Complex64::new(p.g_nuc, 0.0),
        Complex64::zero(),
        Complex64::new(p.g_nuc, 0.0),
        Complex64::new(p.det_nuc, 0.0),
    ];

    let minus_i = Complex64::new(0.0, -1.0);
    let mut comm = [Complex64::zero(); 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut h_rho = Complex64::zero();
            let mut rho_h = Complex64::zero();
            for k in 0..3 {
                h_rho = h_rho + h[i * 3 + k] * rho[k * 3 + j];
                rho_h = rho_h + rho[i * 3 + k] * h[k * 3 + j];
            }
            comm[i * 3 + j] = (h_rho - rho_h) * minus_i;
        }
    }

    // L₁ = |0><1|: L₁ρL₁† = ρ₁₁|0><0|, L₁†L₁ = |1><1|.
    let mut l1 = [Complex64::zero(); 9];
    l1[0] = rho[4];
    for j in 0..3 {
        l1[3 + j] = l1[3 + j] - scale(rho[3 + j], 0.5);
        l1[j * 3 + 1] = l1[j * 3 + 1] - scale(rho[j * 3 + 1], 0.5);
    }

    // L₂ = |1><2|: L₂ρL₂† = ρ₂₂|1><1|, L₂†L₂ = |2><2|.
    let mut l2 = [Complex64::zero(); 9];
    l2[4] = rho[8];
    for j in 0..3 {
        l2[6 + j] = l2[6 + j] - scale(rho[6 + j], 0.5);
        l2[j * 3 + 2] = l2[j * 3 + 2] - scale(rho[j * 3 + 2], 0.5);
    }

    let mut deriv = [Complex64::zero(); 9];
    for i in 0..9 {
        deriv[i] = comm[i] + scale(l1[i], p.gamma_1) + scale(l2[i], p.gamma_2);
    }
    deriv
}

/// One classical RK4 step of size `dt` from time `t`.
pub fn rk4_step(rho: &Rho3, t: f64, dt: f64, p: &LindbladParams) -> Rho3 {
    let advance = |k: &Rho3, h: f64| {
        let mut out = [Complex64::zero(); 9];
        for i in 0..9 {
            out[i] = rho[i] + scale(k[i], h);
        }
        out
    };

    let k1 = compute_lindblad_derivative(rho, t, p);
    let k2 = compute_lindblad_derivative(&advance(&k1, 0.5 * dt), t + 0.5 * dt, p);
    let k3 = compute_lindblad_derivative(&advance(&k2, 0.5 * dt), t + 0.5 * dt, p);
    let k4 = compute_lindblad_derivative(&advance(&k3, dt), t + dt, p);

    let mut next = [Complex64::zero(); 9];
    for i in 0..9 {
        let weighted = k1[i] + scale(k2[i], 2.0) + scale(k3[i], 2.0) + k4[i];
        next[i] = rho[i] + scale(weighted, dt / 6.0);
    }
    next
}

/// Result of an adaptive RK4 step.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveStep {
    /// Accepted state after advancing by `dt_taken`.
    pub rho: Rho3,
    /// Step actually taken.
    pub dt_taken: f64,
    /// Suggested step size for the next call.
    pub dt_next: f64,
    /// Richardson error estimate (max-norm) of the accepted step.
    pub error: f64,
}

/// Adaptive RK4 via step doubling: one full step is compared against two half
/// steps and the step size is shrunk until the Richardson error estimate falls
/// below `tol`. The returned state is the Richardson-extrapolated (5th-order)
/// two-half-step solution.
pub fn rk4_adaptive_step(
    rho: &Rho3,
    t: f64,
    dt: f64,
    tol: f64,
    dt_min: f64,
    p: &LindbladParams,
) -> AdaptiveStep {
    const SAFETY: f64 = 0.9;
    const MAX_GROWTH: f64 = 4.0;
    const MIN_SHRINK: f64 = 0.1;

    let mut h = dt;
    loop {
        let full = rk4_step(rho, t, h, p);
        let half = rk4_step(rho, t, 0.5 * h, p);
        let two_half = rk4_step(&half, t + 0.5 * h, 0.5 * h, p);

        let mut error = 0.0_f64;
        let mut out = [Complex64::zero(); 9];
        for i in 0..9 {
            let diff = two_half[i] - full[i];
            error = error.max(diff.norm_sq().sqrt());
            out[i] = two_half[i] + scale(diff, 1.0 / 15.0);
        }

        let ratio = if error > 0.0 {
            SAFETY * (tol / error).powf(0.2)
        } else {
            MAX_GROWTH
        };

        if error <= tol || h <= dt_min {
            let dt_next = (h * ratio.clamp(MIN_SHRINK, MAX_GROWTH)).max(dt_min);
            return AdaptiveStep {
                rho: out,
                dt_taken: h,
                dt_next,
                error,
            };
        }
        h = (h * ratio.max(MIN_SHRINK)).max(dt_min);
    }
}

/// Balance-of-plant power reconciliation (update-10.1 §3).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BopBalance {
    /// Hydraulic pump shaft power `Q·ΔP/η_pump` [W].
    pub p_pump: f64,
    /// Lattice thermal load [W].
    pub q_lattice: f64,
    /// Total heat rejected by the chiller `Q_lattice + P_pump` [W].
    pub q_total: f64,
    /// Compressor electrical draw `Q_total / COP` [W].
    pub p_compressor: f64,
    /// Chiller coefficient of performance.
    pub cop: f64,
}

impl BopBalance {
    /// Combined BOP parasitic electrical load `P_pump + P_compressor` [W].
    pub fn p_parasitic(&self) -> f64 {
        self.p_pump + self.p_compressor
    }
}

/// `P_pump = Q·ΔP/η`, `Q_total = Q_lattice + P_pump`, `P_comp = Q_total/COP`.
pub fn solve_bop_balance(
    q_flow: f64,
    delta_p: f64,
    eta_pump: f64,
    q_lattice: f64,
    cop: f64,
) -> BopBalance {
    let p_pump = (q_flow * delta_p) / eta_pump;
    let q_total = q_lattice + p_pump;
    let p_compressor = q_total / cop;
    BopBalance {
        p_pump,
        q_lattice,
        q_total,
        p_compressor,
        cop,
    }
}

/// Hydraulic operating point of the coolant loop.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BopParams {
    /// Volumetric flow rate [m³/s].
    pub q_flow: f64,
    /// Loop pressure drop [Pa].
    pub delta_p: f64,
    /// Pump efficiency.
    pub eta_pump: f64,
    /// Lattice thermal load [W].
    pub q_lattice: f64,
    /// Chiller COP.
    pub cop: f64,
}

/// Output of [`run_simulation`].
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct SimulationResult {
    /// Density matrix after the final step.
    pub final_rho: Rho3,
    /// Reconciled BOP power budget.
    pub bop: BopBalance,
}

/// Integrates the master equation for `steps` fixed RK4 steps of `dt` from
/// `|0><0|`, then reconciles the BOP power budget.
pub fn run_simulation(
    steps: usize,
    dt: f64,
    lindblad: &LindbladParams,
    bop: &BopParams,
) -> SimulationResult {
    let mut rho = ground_state();
    let mut t = 0.0;
    for _ in 0..steps {
        rho = rk4_step(&rho, t, dt, lindblad);
        t += dt;
    }
    let bop = solve_bop_balance(
        bop.q_flow,
        bop.delta_p,
        bop.eta_pump,
        bop.q_lattice,
        bop.cop,
    );
    SimulationResult {
        final_rho: rho,
        bop,
    }
}

/// C-ABI entry point for [`run_simulation`].
///
/// Returns `0` on success, `-1` if any output pointer is null.
///
/// # Safety
/// `out_rho_re` and `out_rho_im` must point to 9 writable `f64`s; the three
/// scalar outputs must each point to one writable `f64`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn shbt_cf_run_coupled_simulation(
    steps: usize,
    dt: f64,
    omega_0: f64,
    omega_l: f64,
    delta: f64,
    g_nuc: f64,
    det_nuc: f64,
    gamma_1: f64,
    gamma_2: f64,
    q_flow: f64,
    delta_p: f64,
    eta_pump: f64,
    q_lattice: f64,
    cop: f64,
    out_rho_re: *mut f64,
    out_rho_im: *mut f64,
    out_p_pump: *mut f64,
    out_q_total: *mut f64,
    out_p_compressor: *mut f64,
) -> i32 {
    if out_rho_re.is_null()
        || out_rho_im.is_null()
        || out_p_pump.is_null()
        || out_q_total.is_null()
        || out_p_compressor.is_null()
    {
        return -1;
    }

    let lindblad = LindbladParams {
        omega_0,
        omega_l,
        delta,
        g_nuc,
        det_nuc,
        gamma_1,
        gamma_2,
    };
    let bop = BopParams {
        q_flow,
        delta_p,
        eta_pump,
        q_lattice,
        cop,
    };
    let result = run_simulation(steps, dt, &lindblad, &bop);

    for (i, z) in result.final_rho.iter().enumerate() {
        *out_rho_re.add(i) = z.re;
        *out_rho_im.add(i) = z.im;
    }
    *out_p_pump = result.bop.p_pump;
    *out_q_total = result.bop.q_total;
    *out_p_compressor = result.bop.p_compressor;

    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> LindbladParams {
        LindbladParams {
            omega_0: 0.8,
            omega_l: 1.3,
            delta: 0.2,
            g_nuc: 0.5,
            det_nuc: -0.1,
            gamma_1: 0.05,
            gamma_2: 0.02,
        }
    }

    fn trace(rho: &Rho3) -> Complex64 {
        rho[0] + rho[4] + rho[8]
    }

    #[test]
    fn bop_matches_master_power_table() {
        let bop = solve_bop_balance(2.0833e-4, 85_000.0, 0.65, 147.66, 3.0);
        assert!((bop.p_pump - 27.2432).abs() < 1e-3);
        assert!((bop.q_total - 174.9032).abs() < 1e-3);
        assert!((bop.p_compressor - 58.3011).abs() < 1e-3);
        assert!((bop.p_parasitic() - 85.5443).abs() < 1e-3);
    }

    #[test]
    fn derivative_is_traceless_and_hermitian() {
        let mut rho = ground_state();
        rho[1] = Complex64::new(0.1, 0.2);
        rho[3] = Complex64::new(0.1, -0.2);
        rho[0] = Complex64::new(0.7, 0.0);
        rho[4] = Complex64::new(0.2, 0.0);
        rho[8] = Complex64::new(0.1, 0.0);
        let d = compute_lindblad_derivative(&rho, 0.37, &params());
        let tr = trace(&d);
        assert!(tr.re.abs() < 1e-14 && tr.im.abs() < 1e-14);
        for i in 0..3 {
            for j in 0..3 {
                let a = d[i * 3 + j];
                let b = d[j * 3 + i];
                assert!((a.re - b.re).abs() < 1e-14 && (a.im + b.im).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn rk4_preserves_trace_and_populates_excited_states() {
        let p = params();
        let mut rho = ground_state();
        let dt = 1e-3;
        for step in 0..5000 {
            rho = rk4_step(&rho, step as f64 * dt, dt, &p);
        }
        let tr = trace(&rho);
        assert!((tr.re - 1.0).abs() < 1e-9 && tr.im.abs() < 1e-12);
        assert!(rho[4].re > 0.0 && rho[8].re > 0.0);
        for k in [0, 4, 8] {
            assert!(rho[k].re >= -1e-12 && rho[k].re <= 1.0 + 1e-12);
        }
    }

    #[test]
    fn adaptive_step_meets_tolerance_and_matches_fixed_step() {
        let p = params();
        let rho0 = ground_state();
        let tol = 1e-10;
        let mut t = 0.0;
        let mut dt: f64 = 0.05;
        let mut rho = rho0;
        let t_end = 1.0;
        while t < t_end - 1e-15 {
            let h = dt.min(t_end - t);
            let step = rk4_adaptive_step(&rho, t, h, tol, 1e-6, &p);
            assert!(step.error <= tol);
            rho = step.rho;
            t += step.dt_taken;
            dt = step.dt_next;
        }
        let mut reference = rho0;
        let n = 20_000;
        let h = t_end / n as f64;
        for i in 0..n {
            reference = rk4_step(&reference, i as f64 * h, h, &p);
        }
        for i in 0..9 {
            assert!((rho[i] - reference[i]).norm_sq().sqrt() < 1e-7);
        }
    }

    #[test]
    fn c_abi_round_trip() {
        let mut re = [0.0; 9];
        let mut im = [0.0; 9];
        let (mut pp, mut qt, mut pc) = (0.0, 0.0, 0.0);
        let status = unsafe {
            shbt_cf_run_coupled_simulation(
                1000,
                1e-3,
                0.8,
                1.3,
                0.2,
                0.5,
                -0.1,
                0.05,
                0.02,
                2.0833e-4,
                85_000.0,
                0.65,
                147.66,
                3.0,
                re.as_mut_ptr(),
                im.as_mut_ptr(),
                &mut pp,
                &mut qt,
                &mut pc,
            )
        };
        assert_eq!(status, 0);
        assert!((re[0] + re[4] + re[8] - 1.0).abs() < 1e-9);
        assert!((pc - 58.3011).abs() < 1e-3);
        assert!((pp - 27.2432).abs() < 1e-3);
        assert!((qt - 174.9032).abs() < 1e-3);

        let status = unsafe {
            shbt_cf_run_coupled_simulation(
                1,
                1e-3,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                1.0,
                std::ptr::null_mut(),
                im.as_mut_ptr(),
                &mut pp,
                &mut qt,
                &mut pc,
            )
        };
        assert_eq!(status, -1);
    }
}
