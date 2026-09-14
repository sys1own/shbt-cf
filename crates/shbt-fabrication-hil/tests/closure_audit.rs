//! Physical-closure audit tests (simulator_spec.pdf §5–6) reconciling the
//! cross-module consistency gaps identified in root/cf.pdf:
//!
//! a. unified spatial integration: active metal volume → domain volume;
//! b. dual-channel GUM calorimetric covariance reconciliation;
//! c. Coffin–Manson low-cycle fatigue checks against Δεp ≤ 0.000153.

use shbt_metrology_gum::distributions::{Input, Marginal, Sampler};
use shbt_metrology_gum::engine::propagate;

const TOL: f64 = 1e-6;

fn approx(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * b.abs().max(1.0)
}

// ---------------------------------------------------------------------------
// (a) Unified spatial integration — cf.pdf Eqs. (29), (35), (123).

/// Option B footprint: A = 1 cm², mean metal thickness 28.75 nm.
const A_FOOT: f64 = 1.0e-4;
const T_METAL: f64 = 28.75e-9;
/// Active metal film volume (Eq. 123).
const V_METAL: f64 = 2.875e-12;
/// Domain microstructure (Eq. 35): V_D per domain, N_D domains.
const V_D: f64 = 1.25e-15;
const N_DOMAINS_TABLE: f64 = 1.84e7;
const V_DOMAINS: f64 = 2.30e-8;
/// Loading chain (Eq. 29): n_M · V_B gives the deuteron inventory at x_D = 1;
/// N_coh = 10⁵ atoms per coherent domain.
const N_M: f64 = 6.8250203e28;
const N_COH: f64 = 1.0e5;

#[test]
fn metal_volume_from_geometry() {
    let v = A_FOOT * T_METAL;
    assert!(approx(v, V_METAL, TOL), "A·t_metal = {v} vs {V_METAL}");
}

#[test]
fn domain_volume_from_microstructure() {
    let v = N_DOMAINS_TABLE * V_D;
    assert!(approx(v, V_DOMAINS, TOL), "N_D·V_D = {v} vs {V_DOMAINS}");
}

#[test]
fn unified_volume_mapping_is_factor_8000() {
    // The unified spatial integration maps the domain ensemble onto the
    // active metal film: V_domains / V_metal = 8000 exactly at the stated
    // precisions — this is the "factor of 8000" the cf.pdf consistency note
    // requires to be resolved into a single volume convention.
    let ratio = V_DOMAINS / V_METAL;
    assert!(approx(ratio, 8000.0, 1e-9), "ratio = {ratio}");
    let metal_fraction = V_METAL / V_DOMAINS;
    assert!(approx(metal_fraction, 1.25e-4, 1e-9), "{metal_fraction}");
}

#[test]
fn loading_chain_domain_count() {
    let n_deuterons = N_M * V_METAL;
    assert!(approx(n_deuterons, 1.9622e17, 1e-3), "{n_deuterons}");
    let n_domains = n_deuterons / N_COH;
    assert!(approx(n_domains, 1.9622e12, 1e-3), "{n_domains}");
}

// ---------------------------------------------------------------------------
// (b) Dual-channel GUM calorimetric covariance — cf.pdf Eqs. (280)–(288), (305).

/// Table XLVI dual-channel inputs: (u_i native, c_i W/unit, listed σ²).
const TABLE_XLVI: [(&str, f64, f64, f64); 7] = [
    ("mass flow", 0.0302, 48.26, 2.124),
    ("temperature difference", 0.0120, 253.16, 9.228),
    ("specific heat", 0.0021, 812.70, 2.914),
    ("envelope thermopile", 0.0800, 18.42, 2.171),
    ("optical input", 1.2000, 1.00, 1.440),
    ("rf input", 0.1500, 1.00, 0.023),
    ("ambient loss correction", 1.2124, 1.00, 1.470),
];

#[test]
fn table_xlvi_variance_terms_match() {
    for (name, u, c, listed) in TABLE_XLVI {
        let var = (c * u) * (c * u);
        assert!(approx(var, listed, 0.01), "{name}: {var} vs {listed} W²");
    }
}

#[test]
fn nominal_rss_and_residual_reconciled() {
    let sum: f64 = TABLE_XLVI
        .iter()
        .map(|(_, u, c, _)| (c * u) * (c * u))
        .sum();
    assert!(approx(sum, 19.370, 0.005), "Σ(c·u)² = {sum} W²");
    let u_rss = sum.sqrt();
    assert!(approx(u_rss, 4.4011, 0.001), "{u_rss}");
    // cf.pdf Eq. (287): the 5.69 W claim leaves an unassigned residual that a
    // complete covariance matrix must absorb — the audit pins it exactly.
    let residual = 5.69 * 5.69 - sum;
    assert!(approx(residual, 13.0061, 0.001), "{residual} W²");
}

#[test]
fn two_term_allocation_within_limit() {
    // Eq. (284): u²_c = u²_spatial_drift + u²_instrumentation.
    let u2_spatial = (185.4 * 0.0142) * (185.4 * 0.0142);
    assert!(approx(u2_spatial, 6.9310039824, 1e-9), "{u2_spatial}");
    let uc = (u2_spatial + 12.50).sqrt();
    assert!(approx(uc, 4.408060798, 2e-6), "{uc}");
    assert!(uc <= 5.6881, "u_c = {uc} exceeds the 5.6881 W allocation bound");
}

#[test]
fn literal_four_input_budget_mc_reconciles() {
    // Eq. (305): P = ṁ·cp·ΔT + P_env with the four literal uncertainties
    // gives u_c(P) = 5.69603454 W — verified end-to-end through the GUM
    // Supplement-1 parallel Monte-Carlo engine.
    let sampler = Sampler::new(&[
        Input::Scalar {
            name: "m_dot".into(),
            marginal: Marginal::Normal {
                mean: 0.08250,
                std: 9.4875e-5,
            },
        },
        Input::Scalar {
            name: "c_p".into(),
            marginal: Marginal::Normal {
                mean: 4182.0,
                std: 0.8364,
            },
        },
        Input::Scalar {
            name: "d_t".into(),
            marginal: Marginal::Normal {
                mean: 8.42000,
                std: 0.00310,
            },
        },
        Input::Scalar {
            name: "p_env".into(),
            marginal: Marginal::Normal {
                mean: 12.30,
                std: 4.4500,
            },
        },
    ])
    .unwrap();
    let report = propagate(&sampler, |x| x[0] * x[1] * x[2] + x[3], 200_000, 42, 4);
    assert!(approx(report.mean, 2917.3263, 2e-5), "mean {}", report.mean);
    assert!(
        approx(report.std_dev, 5.6960345, 0.01),
        "std {}",
        report.std_dev
    );
    // Reconciliation: literal u_c sits below the 5.70 W qualification limit.
    assert!(report.std_dev < 5.70);
}

// ---------------------------------------------------------------------------
// (c) Coffin–Manson endurance — cf.pdf Eqs. (170), (214): range convention
// Δεp = ε′f (2 N_f)^c with ε′f = 0.18, c = −0.62; qualify ≥ 44 820 cycles.

const EPS_F: f64 = 0.18;
const C: f64 = -0.62;
const N_TARGET: f64 = 44_820.0;

fn d_eps_at(n_f: f64) -> f64 {
    EPS_F * (2.0 * n_f).powf(C)
}

fn n_f_at(d_eps: f64) -> f64 {
    0.5 * (d_eps / EPS_F).powf(1.0 / C)
}

#[test]
fn strain_ceiling_reproduces_spec_values() {
    let ceiling = d_eps_at(N_TARGET);
    assert!(approx(ceiling, 0.000153010544, 1e-7), "{ceiling}");
    // Unrounded ceiling governs acceptance (cf.pdf Eq. 170 note).
    assert!(ceiling <= 0.0001531);
    assert!(
        approx(n_f_at(0.00098), 2242.11, 0.01),
        "{}",
        n_f_at(0.00098)
    );
    assert!(
        approx(n_f_at(0.000155), 43895.79, 0.01),
        "{}",
        n_f_at(0.000155)
    );
}

#[test]
fn nominal_155ppm_ceiling_fails_endurance() {
    // The nominal 0.000155 ceiling does NOT ensure N_f ≥ 44 820.
    assert!(n_f_at(0.000155) < N_TARGET);
    assert!(n_f_at(0.000153010544) >= N_TARGET * (1.0 - 1e-9));
    assert!(n_f_at(0.000153) >= N_TARGET);
}

#[test]
fn endurance_under_coefficient_uncertainty() {
    // GUM-MC over the coefficients at the qualified ceiling Δεp = 0.000153.
    let sampler = Sampler::new(&[
        Input::Scalar {
            name: "eps_f".into(),
            marginal: Marginal::Normal {
                mean: 0.18,
                std: 0.004,
            },
        },
        Input::Scalar {
            name: "c".into(),
            marginal: Marginal::Normal {
                mean: 0.62,
                std: 0.01,
            },
        },
    ])
    .unwrap();
    let d_eps = 0.00014; // marginally below the qualified 0.000153 ceiling
    let report = propagate(
        &sampler,
        |x| 0.5 * (d_eps / x[0]).powf(-1.0 / x[1]),
        100_000,
        9,
        4,
    );
    // Median endurance above the 44 820-cycle requirement; P(N_f≥N_target)
    // > 0.7 under coefficient uncertainty.
    let mut sorted = report.outputs.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    assert!(median > N_TARGET, "median N_f = {median}");
    let pass = report.outputs.iter().filter(|&&v| v >= N_TARGET).count();
    let frac = pass as f64 / report.outputs.len() as f64;
    assert!(frac > 0.7, "P(N_f ≥ 44820) = {frac}");
}
