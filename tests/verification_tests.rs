use shbt_cf::run_simulation;

#[test]
fn verifies_driven_screening_shift() {
    let summary = run_simulation();
    assert!(
        (summary.u_eff - 350.0).abs() <= 0.5,
        "U_eff = {} eV",
        summary.u_eff
    );
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
        (summary.p_net - 31.76).abs() <= 1e-6,
        "P_net = {} W",
        summary.p_net
    );
}

#[test]
fn verifies_lattice_branching_ratio() {
    let summary = run_simulation();
    assert!(summary.b_lat > 0.999999, "B_lat = {}", summary.b_lat);
}
