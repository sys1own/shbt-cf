// file: src/physics/metrology.rs

//! Multi-variable GUM (ISO/IEC Guide 98-3) covariance architecture for the
//! plant metrology chain (cf3 spec §5): the eight-input covariance `ΣX`,
//! forward-mode Jacobian `J`, and output covariance `ΣY = J ΣX Jᵀ` for the
//! calorimetry / QMS isotopic / EJ-301 PSD measurands.
//!
//! `X = [Q (L/min), T_in (K), T_out (K), UA_loss (W/K),
//!       I_4He (A), S_4He (A/mbar), Q_tail (pC), Q_total (pC)]`
//! `Y = [P_thermal (W), P_4He (mbar), PSD_FOM]`.

/// Number of metrology inputs.
pub const DIM_X: usize = 8;
/// Number of propagated outputs.
pub const DIM_Y: usize = 3;

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
    0.0185,  // Q
    0.10,    // T_in
    0.10,    // T_out
    0.60,    // UA_loss
    1.9e-12, // I_4He
    5.0e-6,  // S_4He
    0.16,    // Q_tail
    0.30,    // Q_total
];

/// `P_thermal` nominal [W].
pub const P_THERMAL_NOMINAL_W: f64 = 3093.44;
/// `P_4He` nominal [mbar].
pub const P_4HE_NOMINAL_MBAR: f64 = 1.521e-7;
/// EJ-301 PSD figure of merit at the nominal tail/total ratio.
pub const PSD_FOM_NOMINAL: f64 = 2.150;

const RHO_SEC: f64 = 960.6;
const CP_SEC: f64 = 4180.0;
const ETA_CAL: f64 = 0.0301;
const PSD_FOM_SCALE: f64 = 13.405;

/// Full `8×8` input covariance `ΣX` with `r(T_in,T_out)=0.85`,
/// `r(Q_tail,Q_total)=0.92`, `r(Q,UA_loss)=0.30`.
pub fn input_covariance() -> [[f64; DIM_X]; DIM_X] {
    let mut sigma = [[0.0f64; DIM_X]; DIM_X];
    for (i, &u) in INPUT_STD.iter().enumerate() {
        sigma[i][i] = u * u;
    }
    let mut set_corr = |a: usize, b: usize, r: f64| {
        sigma[a][b] = r * INPUT_STD[a] * INPUT_STD[b];
        sigma[b][a] = sigma[a][b];
    };
    set_corr(1, 2, 0.85);
    set_corr(6, 7, 0.92);
    set_corr(0, 3, 0.30);
    sigma
}

/// Measurand vector `Y(X)`.
pub fn measurand(x: &[f64; DIM_X]) -> [f64; DIM_Y] {
    let q_m3_s = x[0] * 1e-3 / 60.0;
    let p_thermal = q_m3_s * RHO_SEC * CP_SEC * ETA_CAL * (x[2] - x[1]);
    let p_4he = x[4] / x[5];
    let psd = x[6] / x[7] * PSD_FOM_SCALE;
    [p_thermal, p_4he, psd]
}

/// Forward-mode Jacobian `J` (rows: outputs; columns: inputs) by analytic
/// partial derivatives of the measurand vector.
pub fn jacobian(x: &[f64; DIM_X]) -> [[f64; DIM_X]; DIM_Y] {
    let mut j = [[0.0f64; DIM_X]; DIM_Y];
    let q_m3_s = x[0] * 1e-3 / 60.0;
    let gain = RHO_SEC * CP_SEC * ETA_CAL;
    let delta_t = x[2] - x[1];
    j[0][0] = (1e-3 / 60.0) * gain * delta_t; // dP/dQ
    j[0][1] = -q_m3_s * gain; // dP/dT_in
    j[0][2] = q_m3_s * gain; //  dP/dT_out
    j[1][4] = 1.0 / x[5]; // dP_4He/dI
    j[1][5] = -x[4] / (x[5] * x[5]); // dP_4He/dS
    j[2][6] = PSD_FOM_SCALE / x[7]; // dFOM/dQ_tail
    j[2][7] = -PSD_FOM_SCALE * x[6] / (x[7] * x[7]); // dFOM/dQ_total
    j
}

/// Output covariance `ΣY = J ΣX Jᵀ`.
pub fn output_covariance() -> [[f64; DIM_Y]; DIM_Y] {
    let sigma_x = input_covariance();
    let j = jacobian(&INPUT_MEANS);
    let mut sigma_y = [[0.0f64; DIM_Y]; DIM_Y];
    for (i, ji) in j.iter().enumerate() {
        for (k, jk) in j.iter().enumerate() {
            let mut acc = 0.0;
            for (a, &jia) in ji.iter().enumerate() {
                let mut inner = 0.0;
                for (b, &jkb) in jk.iter().enumerate() {
                    inner += sigma_x[a][b] * jkb;
                }
                acc += jia * inner;
            }
            sigma_y[i][k] = acc;
        }
    }
    sigma_y
}

/// Standard uncertainties `u(y_j) = sqrt(ΣY_jj)`.
pub fn output_std() -> [f64; DIM_Y] {
    let sigma_y = output_covariance();
    [
        sigma_y[0][0].sqrt(),
        sigma_y[1][1].sqrt(),
        sigma_y[2][2].sqrt(),
    ]
}

/// GUM evaluation at the nominal operating point: `(ȳ, u(y))`.
pub fn evaluate_gum() -> ([f64; DIM_Y], [f64; DIM_Y]) {
    (measurand(&INPUT_MEANS), output_std())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gum_nominals_match_spec() {
        let (values, _) = evaluate_gum();
        assert!((values[0] - P_THERMAL_NOMINAL_W).abs() / P_THERMAL_NOMINAL_W < 0.01);
        assert!((values[1] - P_4HE_NOMINAL_MBAR).abs() / P_4HE_NOMINAL_MBAR < 0.01);
        assert!((values[2] - PSD_FOM_NOMINAL).abs() < 0.01);
    }

    #[test]
    fn output_standard_uncertainties_bounded() {
        let (_, u) = evaluate_gum();
        assert!(u[0] > 0.0 && u[0] < 40.0);
        assert!(u[1] > 0.0 && u[1] < 5.0e-9);
        assert!(u[2] > 0.0 && u[2] < 0.1);
    }

    #[test]
    fn jacobian_matches_finite_difference() {
        let j = jacobian(&INPUT_MEANS);
        let eps = 1e-6;
        for (out, j_row) in j.iter().enumerate().take(DIM_Y) {
            for inp in 0..DIM_X {
                if j_row[inp] == 0.0 {
                    continue;
                }
                let mut xp = INPUT_MEANS;
                let mut xm = INPUT_MEANS;
                let h = eps * INPUT_MEANS[inp].abs().max(eps);
                xp[inp] += h;
                xm[inp] -= h;
                let fd = (measurand(&xp)[out] - measurand(&xm)[out]) / (2.0 * h);
                assert!(
                    (fd - j_row[inp]).abs() / j_row[inp].abs() < 1e-4,
                    "out {out} inp {inp}: fd {fd} vs J {}",
                    j_row[inp]
                );
            }
        }
    }
}
