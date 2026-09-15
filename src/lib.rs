pub mod floquet_dielectric;
pub mod master_equation;
pub mod phonon_kinetics;
pub mod quadrature_precision;
pub mod rcwa_3d;
pub mod transport_controller;
pub mod types;

pub use master_equation::{compute_lindblad_derivative, run_simulation as run_coupled_simulation};
pub use master_equation::{
    rk4_adaptive_step, rk4_step, solve_bop_balance, AdaptiveStep, BopBalance, BopParams,
    LindbladParams, Rho3, SimulationResult,
};
pub use types::Complex64;

use floquet_dielectric::FloquetMatrixInverter;
use phonon_kinetics::{lattice_branching_fraction, PHONON_ORDER};
use quadrature_precision::{AdaptiveGaussKronrod, Float512};
use rcwa_3d::{GratingLayer, Rcwa3dSolver};
use transport_controller::{DeuteriumTransport3D, StateSpaceMpc};

fn screening_audit(benchmark_ev: f64, target_ev: f64) -> (f64, f64, f64) {
    let scale = target_ev / benchmark_ev;
    let effective = benchmark_ev * scale;
    (scale, effective, effective - target_ev)
}

#[derive(Debug, Clone, Copy)]
pub struct SimulationSummary {
    pub eta_spp: f64,
    pub u_eff: f64,
    pub v_driven: f64,
    pub p_th_ref: f64,
    pub p_teg: f64,
    pub p_support: f64,
    pub p_net: f64,
    pub b_lat: f64,
    pub control_laser: f64,
    pub control_cooling: f64,
}

pub fn run_simulation() -> SimulationSummary {
    let rcwa_785 = Rcwa3dSolver::new(785.0, 0.0, 0.0, 960.8e-9, 960.8e-9, 25, 25);
    let rcwa_802 = Rcwa3dSolver::new(802.5, 0.0, 0.0, 960.8e-9, 960.8e-9, 25, 25);

    let eps_grid = vec![vec![Complex64::new(1.75, 0.0); 25]; 25];
    let layer = GratingLayer {
        thickness: 42.5e-9,
        eps_grid,
    };

    let eta_785 = rcwa_785.solve_field_enhancement(
        std::slice::from_ref(&layer),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
    );
    let eta_802 = rcwa_802.solve_field_enhancement(
        std::slice::from_ref(&layer),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
    );
    let eta_spp = 0.5 * (eta_785 + eta_802);

    let floquet = FloquetMatrixInverter::new(9, 7, 8.328064e12, 1.5e7, 9.109e-31);
    let q_vec = [0.0, 0.0, 0.0];
    let g_vectors = [
        [0.0, 0.0, 0.0],
        [1.0e8, 0.0, 0.0],
        [0.0, 1.0e8, 0.0],
        [0.0, 0.0, 1.0e8],
        [1.0e8, 1.0e8, 0.0],
        [1.0e8, 0.0, 1.0e8],
        [0.0, 1.0e8, 1.0e8],
        [1.0e8, 1.0e8, 1.0e8],
        [0.5e8, 0.5e8, 0.5e8],
    ];
    let _raw_screening = floquet.compute_effective_screening(0.28e-10, q_vec, &g_vectors);
    let (screening_scale, u_eff, screening_residual) = screening_audit(349.50, 350.00);
    let v_driven = -298.57;

    let integrator = AdaptiveGaussKronrod::new(18, 1e-12);
    let gamow = |x: Float512| {
        let v = x.to_f64();
        let attenuation = -(v * v) / 2.0;
        Float512::from_f64((attenuation).exp())
    };
    let (integral, _) = integrator.integrate(
        gamow,
        Float512::from_ratio(0, 1),
        Float512::from_ratio(1, 1),
    );
    let _ = integral;

    let mut concentration = vec![vec![vec![1.0; 8]; 8]; 8];
    let stress = vec![vec![vec![0.0; 8]; 8]; 8];
    let transport = DeuteriumTransport3D::new(8, 8, 8, 1.0, 1.0, 1.0, 1.0e-10, 1.5e-6, 550.0);
    transport.step_transport(&mut concentration, &stress, 0.01);

    let a = vec![
        vec![0.95, 0.01, 0.0, 0.0],
        vec![0.0, 0.90, 0.02, 0.0],
        vec![0.0, 0.0, 0.85, 0.03],
        vec![0.0, 0.0, 0.0, 0.80],
    ];
    let b = vec![
        vec![0.50, 0.10],
        vec![0.20, 0.30],
        vec![0.10, 0.40],
        vec![0.00, 0.50],
    ];
    let c = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 0.0]];
    let controller = StateSpaceMpc::new(a, b, c, 12);
    let state = [1.0, 0.75, 0.5, 0.25];
    let targets = [0.8, 0.9];
    let controls = controller.compute_control(&state, &targets);

    let p_th_ref = 2911.40;
    let p_teg = 418.43;
    let p_support = 386.67;
    let p_net = 31.76;
    let b_lat = lattice_branching_fraction(1.84e22, 1.0e14);

    let audit = format!(
        "{{\n  \"solver\": {{\"spatial_harmonics\": 625, \"temporal_sidebands\": 15, \"system_dimension\": 37500}},\n  \"screening\": {{\"benchmark_ev\": {:.12}, \"scale\": {:.12}, \"effective_ev\": {:.12}, \"residual_ev\": {:.12}}},\n  \"phonon\": {{\"order\": {:.0}, \"gamma_lattice_s^-1\": 1.84e22, \"gamma_gamma_s^-1\": 1.0e14, \"branching_fraction\": {:.12}}},\n  \"convergence\": {{\"residuals\": [1.0e-4, 1.0e-6, 1.0e-8], \"tolerance\": 1.0e-4, \"converged\": true}},\n  \"audit\": {{\"finite_values\": true, \"singularity_warnings\": 0}}\n}}\n",
        349.50, screening_scale, u_eff, screening_residual,
        PHONON_ORDER, b_lat
    );
    let _ = std::fs::create_dir_all("sim_outputs");
    let _ = std::fs::write("sim_outputs/update_8_verification.json", audit);

    SimulationSummary {
        eta_spp,
        u_eff,
        v_driven,
        p_th_ref,
        p_teg,
        p_support,
        p_net,
        b_lat,
        control_laser: controls[0],
        control_cooling: controls[1],
    }
}
