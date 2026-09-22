use shbt_cf::run_simulation;

#[test]
fn verifies_driven_screening_shift() {
    let summary = run_simulation();
    assert!(
        (summary.u_eff - 352.48).abs() <= 0.01,
        "U_eff = {} eV",
        summary.u_eff
    );
}

#[test]
fn verifies_audit_export() {
    let _ = run_simulation();
    let audit = std::fs::read_to_string("sim_outputs/simulation_verification.json").unwrap();
    assert!(audit.contains("\"system_dimension\": 15625"));
    assert!(audit.contains("\"finite_values\": true"));
    assert!(audit.contains("\"converged\": true"));
}

#[test]
fn verifies_all_fifty_gates_pass() {
    let _ = run_simulation();
    let audit = std::fs::read_to_string("sim_outputs/simulation_verification.json").unwrap();
    assert!(audit.contains("\"gates_total\": 50"));
    assert!(audit.contains("\"gates_passed\": 50"));
    for gate in 1..=50 {
        assert!(
            audit.contains(&format!("\"GATE-{gate:02}\"")),
            "missing GATE-{gate:02}"
        );
    }
    assert_eq!(audit.matches("\"status\": \"PASS\"").count(), 50);
}

#[test]
fn verifies_driven_pair_energy() {
    let summary = run_simulation();
    assert!(
        (summary.v_driven + 298.57).abs() <= 0.10,
        "V_driven = {} eV",
        summary.v_driven
    );
}

#[test]
fn verifies_net_export_power() {
    let summary = run_simulation();
    assert!(
        (summary.p_net - 555.03).abs() <= 1e-2,
        "P_net = {} W",
        summary.p_net
    );
}

#[test]
fn verifies_lattice_branching_ratio() {
    let summary = run_simulation();
    assert!(summary.b_lat > 0.999999, "B_lat = {}", summary.b_lat);
}
