#![allow(missing_docs)]

use shbt_contact::{BellevilleWasherGeometry, ContactMechanicsSolver};

fn geometry() -> BellevilleWasherGeometry {
    BellevilleWasherGeometry {
        outer_radius: 0.02,
        inner_radius: 0.01,
        thickness: 0.0005,
        initial_cone_angle_rad: 0.12,
        thermal_expansion_coeff: 12e-6,
        youngs_modulus_293k: 200e9,
        youngs_modulus_temperature_slope: 0.0005,
        poissons_ratio: 0.3,
        contact_clearance: 1e-6,
        node_count: 9,
    }
}

#[test]
fn pdas_converges_at_reference_temperature() {
    let solver = ContactMechanicsSolver::new();
    let shape = geometry();
    let mut state = solver.initialize_pdas_solver(&shape);
    let displacement = solver.solve_step_newton_raphson(&shape, &mut state, &[293.0], 10.0);
    assert!(state.converged);
    assert_eq!(displacement.len(), shape.node_count);
    assert!(state.active_indices.len() <= 2);
}

#[test]
fn thermal_delta_changes_modulus_and_displacement() {
    let solver = ContactMechanicsSolver::new();
    let shape = geometry();
    let mut cold = solver.initialize_pdas_solver(&shape);
    let cold_displacement = solver.solve_step_newton_raphson(&shape, &mut cold, &[293.0], 10.0);
    let mut hot = solver.initialize_pdas_solver(&shape);
    let hot_displacement = solver.solve_step_newton_raphson(&shape, &mut hot, &[850.0], 10.0);
    assert!(shape.youngs_modulus(850.0) < shape.youngs_modulus(293.0));
    assert!(hot_displacement[shape.node_count - 1] > cold_displacement[shape.node_count - 1]);
    assert!(hot.converged);
}

#[test]
fn tangent_stiffness_is_positive() {
    let solver = ContactMechanicsSolver::new();
    let stiffness = solver.tangent_stiffness(&geometry(), 850.0, 0.1);
    assert!(stiffness.diag().iter().all(|value| *value > 0.0));
}
