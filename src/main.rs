use shbt_cf::run_simulation;

fn main() {
    let summary = run_simulation();

    println!("SHBT-CF Integrated Simulation Pipeline");
    println!("===================================");
    println!("SPP field enhancement eta_SPP: {:.6}", summary.eta_spp);
    println!("Effective screening U_eff: {:.2} eV", summary.u_eff);
    println!(
        "Screened deuteron pair energy V_driven: {:.2} eV",
        summary.v_driven
    );
    println!("Gamow tunneling integral: evaluated with Float512 / G7-K15");
    println!("Transport step: 3D diffusion + stress-gradient electromigration");
    println!(
        "MPC action: laser = {:.6}, cooling = {:.6}",
        summary.control_laser, summary.control_cooling
    );
    println!("Energy Balance Table");
    println!("-------------------");
    println!("P_th,ref = {:>8.2} W", summary.p_th_ref);
    println!("P_TEG    = {:>8.2} W", summary.p_teg);
    println!("P_support= {:>8.2} W", summary.p_support);
    println!("P_net    = {:>8.2} W", summary.p_net);
    println!("B_lat    = {:>10.7}", summary.b_lat);
}
