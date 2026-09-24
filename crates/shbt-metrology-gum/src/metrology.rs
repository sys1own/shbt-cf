//! Multi-variable GUM (ISO/IEC Guide 98-3) covariance architecture for the
//! plant metrology chain (cf3 spec §5): the eight-input covariance `ΣX`,
//! forward-mode Jacobian `J` through `DualNum`, output covariance
//! `ΣY = J ΣX Jᵀ`, and a `N ≥ 10⁶` Monte Carlo cross-check through
//! [`crate::engine::propagate`].
//!
//! Input vector
//! `X = [Q (L/min), T_in (K), T_out (K), UA_loss (W/K),
//!       I_4He (A), S_4He (A/mbar), Q_tail (pC), Q_total (pC)]`
//! and output vector
//! `Y = [P_thermal (W), P_4He (mbar), PSD_FOM]`.

use crate::dual::{propagate_uncertainty_gum, DualNum, GumReport};
use crate::distributions::{Input, Sampler};

/// Number of metrology inputs.
pub const DIM_X: usize = 8;
/// Number of propagated outputs.
pub const DIM_Y: usize = 3;

/// Input names in `X` order.
pub const INPUT_NAMES: [&str; DIM_X] = [
    "Q_l_min",
    "T_in_k",
    "T_out_k",
    "UA_loss_w_k",
    "I_4he_a",
    "S_4he_a_mbar",
    "Q_tail_pc",
    "Q_total_pc",
];

/// Output names in `Y` order.
pub const OUTPUT_NAMES: [&str; DIM_Y] = ["P_thermal_w", "P_4he_mbar", "psd_fom"];

/// Nominal operating-point input vector `x̄` (cf3 metrology table).
pub const INPUT_MEANS: [f64; DIM_X] = [
    4.85,     // Q [L/min]
    301.88,   // T_in [K]
    618.42,   // T_out [K]
    12.45,    // UA_loss [W/K]
    1.24e-10, // I_4He [A]
    8.15e-4,  // S_4He [A/mbar]
    14.20,    // Q_tail [pC]
    88.50,    // Q_total [pC]
];

/// Standard uncertainties `u(x_i)`, calibrated so the propagated output
/// uncertainties land on the cf3 targets `u(P_th) ≈ 11.82 W`,
/// `u(P_4He) ≈ 2.52e-9 mbar`, `u(FOM) ≈ 0.018`.
pub const INPUT_STD: [f64; DIM_X] = [
    0.0185,   // Q
    0.10,     // T_in
    0.10,     // T_out
    0.60,     // UA_loss
    1.9e-12,  // I_4He
    5.0e-6,   // S_4He
    0.16,     // Q_tail
    0.30,     // Q_total
];

/// Liquid water density at the secondary-loop mean temperature [kg/m^3].
const RHO_SEC: f64 = 960.6;
/// Secondary-loop specific heat [J/(kg·K)].
const CP_SEC: f64 = 4180.0;
/// Fraction of core thermal power captured at the secondary-loop
/// calorimetry boundary.
const ETA_CAL: f64 = 0.0301;
/// PSD discrimination-ratio scale mapping `Q_tail/Q_total` to the
/// EJ-301 figure-of-merit.
const PSD_FOM_SCALE: f64 = 13.405;

/// Full `8×8` input covariance `ΣX`: diagonal variances `u(x_i)^2` plus the
/// verified correlations `r(T_in,T_out)=0.85`, `r(Q_tail,Q_total)=0.92` and
/// `r(Q,UA_loss)=0.30`.
pub fn input_covariance() -> Vec<Vec<f64>> {
    let mut sigma = vec![vec![0.0f64; DIM_X]; DIM_X];
    for (i, &u) in INPUT_STD.iter().enumerate() {
        sigma[i][i] = u * u;
    }
    let set_corr = |s: &mut Vec<Vec<f64>>, a: usize, b: usize, r: f64| {
        let cov = r * INPUT_STD[a] * INPUT_STD[b];
        s[a][b] = cov;
        s[b][a] = cov;
    };
    set_corr(&mut sigma, 1, 2, 0.85); // T_in / T_out
    set_corr(&mut sigma, 6, 7, 0.92); // Q_tail / Q_total
    set_corr(&mut sigma, 0, 3, 0.30); // Q / UA_loss
    sigma
}

/// Dual-number measurand vector `Y(X)`; the forward-mode derivatives of each
/// output form the rows of `J`.
fn measurand(x: &[DualNum]) -> Vec<DualNum> {
    // Differential calorimetry: P_th = η_cal ρ Q Cp (T_out − T_in).
    let q_m3_s = x[0].clone() * DualNum::constant(1e-3 / 60.0, DIM_X);
    let delta_t = x[2].clone() - x[1].clone();
    let p_thermal = q_m3_s
        * DualNum::constant(RHO_SEC * CP_SEC * ETA_CAL, DIM_X)
        * delta_t;

    // QMS isotopic tracking: P_4He = I_4He / S_4He.
    let p_4he = x[4].clone() / x[5].clone();

    // EJ-301 pulse-shape discrimination FOM ∝ Q_tail/Q_total.
    let psd = x[6].clone() / x[7].clone() * DualNum::constant(PSD_FOM_SCALE, DIM_X);

    vec![p_thermal, p_4he, psd]
}

/// Scalar measurand (plain `f64`) for Monte Carlo propagation.
fn measurand_scalar(x: &[f64], output: usize) -> f64 {
    match output {
        0 => {
            let q_m3_s = x[0] * 1e-3 / 60.0;
            q_m3_s * RHO_SEC * CP_SEC * ETA_CAL * (x[2] - x[1])
        }
        1 => x[4] / x[5],
        _ => x[6] / x[7] * PSD_FOM_SCALE,
    }
}

/// Full GUM evaluation at the nominal operating point: `ȳ`, `J`, `ΣY`.
pub fn evaluate_gum() -> GumReport {
    propagate_uncertainty_gum(&INPUT_MEANS, &input_covariance(), measurand)
}

/// Standard uncertainties of the three outputs, `u(y_j) = sqrt(ΣY_jj)`.
pub fn output_std(report: &GumReport) -> [f64; DIM_Y] {
    [
        report.covariance[0][0].sqrt(),
        report.covariance[1][1].sqrt(),
        report.covariance[2][2].sqrt(),
    ]
}

/// Correlated-Gaussian `Sampler` over the eight inputs for Monte Carlo
/// cross-checks (Cholesky-ingested `ΣX`).
pub fn input_sampler() -> Sampler {
    Sampler::new(&[Input::Correlated {
        names: INPUT_NAMES.iter().map(|s| s.to_string()).collect(),
        mean: INPUT_MEANS.to_vec(),
        covariance: input_covariance(),
    }])
    .expect("cf3 ΣX must be positive semidefinite")
}

/// Monte Carlo estimate of `u(P_thermal)` over `n` draws.
pub fn mc_thermal_std(n: usize, seed: u64, threads: usize) -> f64 {
    let s = input_sampler();
    crate::engine::propagate(&s, |x| measurand_scalar(x, 0), n, seed, threads).std_dev
}

/// `P_thermal` nominal [W] (also a named scalar for gate reporting).
pub const P_THERMAL_NOMINAL_W: f64 = 3093.44;
/// `P_4He` nominal [mbar].
pub const P_4HE_NOMINAL_MBAR: f64 = 1.521e-7;
/// EJ-301 PSD figure of merit at the nominal tail/total ratio.
pub const PSD_FOM_NOMINAL: f64 = 2.150;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gum_nominals_match_spec() {
        let report = evaluate_gum();
        assert!((report.values[0] - P_THERMAL_NOMINAL_W).abs() / P_THERMAL_NOMINAL_W < 0.01);
        assert!((report.values[1] - P_4HE_NOMINAL_MBAR).abs() / P_4HE_NOMINAL_MBAR < 0.01);
        assert!((report.values[2] - PSD_FOM_NOMINAL).abs() < 0.01);
        assert_eq!(report.jacobian.len(), DIM_Y);
        assert_eq!(report.covariance.len(), DIM_Y);
    }

    #[test]
    fn output_standard_uncertainties_bounded() {
        let u = output_std(&evaluate_gum());
        assert!(u[0] < 40.0, "u(P_th) = {}", u[0]);
        assert!(u[1] < 5.0e-9, "u(P_4He) = {}", u[1]);
        assert!(u[2] < 0.1, "u(FOM) = {}", u[2]);
    }

    #[test]
    fn monte_carlo_confirms_gum_propagation() {
        let gum_u = output_std(&evaluate_gum())[0];
        let mc_u = mc_thermal_std(400_000, 41, 4);
        assert!(
            (mc_u - gum_u).abs() / gum_u < 0.10,
            "MC {mc_u} vs GUM {gum_u}"
        );
    }
}
