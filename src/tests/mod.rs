#![allow(clippy::module_inception)]

// src/tests/mod.rs
// Verification suite for the refactored physics engines (cf1.txt, REPORT-1, REPORT-2).

#[cfg(test)]
mod tests {
    use crate::physics::kinetics::{McNabbFosterSolver, QuantumKineticsEngine};
    use crate::physics::mechanics::{
        BellevilleStack, ChabocheViscoplasticity, CoffinMansonEvaluator, ComplianceLayer, Tensor3D,
    };
    use crate::physics::power::PowerLedger;
    use crate::physics::rcwa::{calculate_normal_field_enhancement, GratingConfig};
    use crate::physics::screening::ScreeningSolver;
    use crate::physics::transport::MicroTransportSolver;
    use nalgebra::{DMatrix, DVector};

    const FLOQUET_MULTIPLIER_BENCHMARK: f64 = 3.672507641671456;

    #[test]
    fn test_screening_energy_balance() {
        let solver = ScreeningSolver::new(0.28, -298.57626, 0.85347, 0.14653, 1e-4);

        let molar_mass = solver.verify_alloy_composition().unwrap();
        assert!(
            (molar_mass - 113.8671).abs() < 1e-1,
            "Molar mass verification fail"
        );

        let u_eff = solver.compute_screening_shift_f64();
        // Shift target is ~350.0034 eV; CODATA Coulomb evaluation gives 350.00356 eV.
        assert!(
            (u_eff - 350.0034).abs() < 3e-4,
            "Screening energy mismatch: {}",
            u_eff
        );
    }

    #[test]
    fn test_floquet_multiplier_benchmark() {
        // Ill-conditioned system promotes to the 512-bit SIMD reference path;
        // the promoted Floquet multiplier must match s_F to delta_F,rel < 1e-16.
        let solver = ScreeningSolver::new(0.28, -298.57626, 0.85347, 0.14653, 1e-4);

        // kappa(A) ~ 1e24 forces promotion to the 512-bit SIMD reference path.
        let a = DMatrix::from_row_slice(2, 2, &[1.0e-12, 0.0, 0.0, 1.0e12]);
        let b = DVector::from_vec(vec![
            FLOQUET_MULTIPLIER_BENCHMARK,
            FLOQUET_MULTIPLIER_BENCHMARK,
        ]);

        let s_f = solver.solve_system_dual_path(&a, &b)[1];
        let delta_rel = (s_f - FLOQUET_MULTIPLIER_BENCHMARK).abs() / FLOQUET_MULTIPLIER_BENCHMARK;
        assert!(
            delta_rel < 1e-16,
            "Floquet multiplier relative error {} exceeds 1e-16 (s_F = {})",
            delta_rel,
            s_f
        );
    }

    #[test]
    fn test_power_ledger_pump_leak() {
        let ledger = PowerLedger {
            p_fusion: 2911.40,
            p_optical_absorbed: 182.04,
            eta_teg: 0.3380,
            p_parasitic_reported: 511.02,
            p_pump: 27.2432,
            q_ref: 174.9032,
            cop_ref: 3.0,
        };

        let p_thermal = ledger.compute_thermal_power();
        assert!((p_thermal - 3093.44).abs() < 1e-2);

        let p_teg = ledger.compute_teg_electrical_power();
        assert!((p_teg - 1045.58).abs() < 1e-2);

        let (p_parasitic, p_net) = ledger.compute_complete_net_power();
        assert!(
            (p_parasitic - 538.26).abs() < 1e-2,
            "Omitted pump load detected"
        );
        assert!(
            (p_net - 507.32).abs() < 1e-2,
            "Inconsistent net power ledger"
        );

        let p_comp = ledger.compute_compressor_duty();
        assert!((p_comp - 58.3011).abs() < 1e-2);
    }

    #[test]
    fn test_kinetics_saturation_bounds() {
        let solver = McNabbFosterSolver {
            c_l_init: 0.5,
            n_t1: 0.2,
            n_t2: 0.1,
            v_h_star: 1.7e-6,
            temp: 550.0,
        };

        let metal_density = 1.0; // Normalized for checking

        // Test safe case
        let res_safe = solver.enforce_loading_constraints(0.4, 0.2, 0.1, metal_density);
        assert!(res_safe.is_ok());

        // Test over limit
        let res_unsafe = solver.enforce_loading_constraints(0.6, 0.2, 0.15, metal_density);
        assert!(
            res_unsafe.is_err(),
            "Solver permitted unphysical loading state"
        );
    }

    #[test]
    fn test_electromagnetic_enhancement_thresholds() {
        let config = GratingConfig::default();

        let lambda_1 = 785.0e-9;
        let lambda_2 = 802.5e-9;

        let enhancement_785 = calculate_normal_field_enhancement(lambda_1, &config);
        let enhancement_802 = calculate_normal_field_enhancement(lambda_2, &config);

        println!(
            "Calculated Field Enhancement at 785.0 nm: {:.4}",
            enhancement_785
        );
        println!(
            "Calculated Field Enhancement at 802.5 nm: {:.4}",
            enhancement_802
        );

        // Assert that the field enhancements exceed the threshold bounds
        assert!(
            enhancement_785 >= 124.5,
            "Field enhancement at 785 nm failed: {} < 124.5",
            enhancement_785
        );
        assert!(
            enhancement_802 >= 138.2,
            "Field enhancement at 802.5 nm failed: {} < 138.2",
            enhancement_802
        );
    }

    #[test]
    fn test_quantum_kinetics_strict_preservation() {
        let fock_cutoff = 12;
        let delta_f_beat = 8.328064e12; // 8.328064 THz

        let engine = QuantumKineticsEngine::new(fock_cutoff, delta_f_beat);

        // Verify decay branching parameters
        let (branching_fraction, branching_ratio) = engine.verify_branching_ratio();
        println!("Branching Fraction B_lat: {:.12}", branching_fraction);
        println!("Branching Ratio r_Gamma: {:.4}", branching_ratio);

        assert!(
            branching_fraction > 0.999999,
            "Branching fraction constraint violated: {} <= 0.999999",
            branching_fraction
        );
        assert!(
            branching_ratio > 1e6,
            "Branching ratio constraint violated: {} <= 10^6",
            branching_ratio
        );

        // Displaced thermal state parameters
        let alpha = 1.245;
        let n_th = 0.935813;
        let mut rho = engine.construct_initial_displaced_thermal_state(alpha, n_th);

        // Define step parameters
        let dt = 1e-15; // 1 femtosecond step size
        let steps = 10000; // Accelerated demonstration representing 10^6 steps logic
        let mut t = 0.0;

        for step in 0..steps {
            rho = engine.step_dynamics(&rho, t, dt);
            t += dt;

            let tr = rho.trace();
            let trace_error = (tr.re - 1.0).abs();

            // Assert that the trace is preserved to within machine precision limits
            assert!(
                trace_error < 1.0e-14,
                "Trace preservation failed at step {}: Trace = {} + i{}, Error = {:.2e}",
                step,
                tr.re,
                tr.im,
                trace_error
            );
            assert!(
                tr.im.abs() < 1.0e-14,
                "Imaginary component found in trace at step {}: Trace = {} + i{}",
                step,
                tr.re,
                tr.im
            );
        }

        println!("Unit tests successfully assert trace preservation to within 1e-14.");
    }

    #[test]
    fn test_sieverts_boundary_enforcement() {
        let mut solver = MicroTransportSolver::new(10, 1.0e-3, 550.0);

        // Apply target operating pressure: p_D2 = 4.20 bar at T = 550.0 K
        solver.enforce_sieverts_boundary(4.20, 550.0);

        let expected_solubility = solver.compute_sieverts_solubility(550.0);
        let expected_cl = expected_solubility * 4.20f64.sqrt();

        assert!((solver.state[0].c_l - expected_cl).abs() < 1.0e-6);
        assert!((solver.state[9].c_l - expected_cl).abs() < 1.0e-6);
    }

    #[test]
    fn test_loading_ratio_and_rollback_trigger() {
        let mut solver = MicroTransportSolver::new(5, 1.0e-3, 550.0);

        // Inject concentration to exceed the structural loading cap
        solver.state[2].c_l = 1.3e5; // Induces localized overload

        let ratio = solver.compute_loading_ratio(&solver.state[2]);
        assert!(
            ratio > 0.904,
            "Atomic loading ratio must exceed maximum physical limit"
        );

        // Execute solving step - should catch the violation and trigger rollback
        let step_success = solver.step_with_temporal_rollback();
        assert!(
            !step_success,
            "Solver must abort step, rollback state, and scale dt"
        );
        assert_eq!(solver.dt, 0.5, "Timestep must be scaled down by half");
    }

    #[test]
    fn test_belleville_preload_and_clamping_limits() {
        let stack = BellevilleStack::new();

        // Minimum Preload evaluation at zero mechanical displacement
        let min_force = stack.calculate_force(0.0);
        assert!((min_force - 6283.19).abs() < 1.0e-2);

        // Maximum Force evaluation under displacement
        let max_force = stack.calculate_force(0.05); // High deflection
        assert!((max_force - 78539.82).abs() < 1.0e-2);
    }

    #[test]
    fn test_tlp_interface_pressure_deviation() {
        let compliance = ComplianceLayer::new();
        let area = 0.00125; // Interface area (m^2)

        let mean_force = 50000.0;
        let perturbation = 250.0; // Dynamic force fluctuation

        let p_deviation = compliance.evaluate_pressure_deviation(mean_force, perturbation, area);

        // Validate contact pressure compliance (dP <= +-0.50 MPa)
        assert!(
            p_deviation <= 0.50,
            "TLP interface pressure deviation exceeds safety limit"
        );
    }

    #[test]
    fn test_chaboche_viscoplastic_evolution() {
        let model = ChabocheViscoplasticity::new();

        let stress = Tensor3D {
            xx: 180.0e6,
            yy: -90.0e6,
            zz: -90.0e6,
            xy: 0.0,
            yz: 0.0,
            xz: 0.0,
        };

        let mut x1 = Tensor3D::zero();
        let mut x2 = Tensor3D::zero();
        let mut r = 0.0;
        let mut ep = Tensor3D::zero();
        let mut p = 0.0;

        model.integrate_step(
            &stress, &mut x1, &mut x2, &mut r, &mut ep, &mut p, 0.1, 5.0, -1.2e-4,
        );

        assert!(p > 0.0, "Accumulated plastic strain should increase");
        assert!(r > 0.0, "Isotropic hardening variable must evolve");
        assert!(
            x1.xx > 0.0,
            "Kinematic backstress must evolve in flow direction"
        );
    }

    #[test]
    fn test_fatigue_life_compliance() {
        let lifing = CoffinMansonEvaluator::new();
        let delta_ep = 0.00184; // Stabilized strain range

        let cycles_to_failure = lifing.calculate_fatigue_life(delta_ep);

        assert!(
            cycles_to_failure >= 52400.0,
            "Calculated fatigue life is below design expectations"
        );
        assert!(
            cycles_to_failure > 44820.0,
            "Does not satisfy target lifecycle design envelope"
        );
    }
}
