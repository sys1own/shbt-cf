//! Inverse Floquet–Adler–Wiser non-local dielectric solver
//! (cf.pdf §"Floquet dielectric response", simulator_spec.pdf §2–§3).
//!
//! Assembles the population-diagonal Floquet polarizability
//!
//! ```text
//! Π_{GG'}^{mn}(q,ω) = (1/Ω_cell) Σ_{k,ab,p} w_k (f_ak − f_{b,k+q})
//!     · M_{m−p,G}^{ab} · [M_{n−p,G'}^{ab}]* / (ε_ak − ε_{b,k+q} + ℏω + pℏΩ_D + iη)
//! ```
//!
//! forms `ε_{GG'}^{mn} = δ_{GG'}δ_{mn} − v_G Π_{GG'}^{mn}` with
//! `v_G = e²/(ε₀|q+G|²)`, and inverts the composite `(G,m)` matrix with a
//! condition-number check (`shbt_core_math::precision`). Iterative unit-cell
//! refinement converges when `‖ε⁻¹_{k+1} − ε⁻¹_k‖∞ < 1e-14`.
//!
//! Microscopic inputs (band energies, occupations, Floquet matrix elements
//! `M_{ℓ,G}^{ab}`) are supplied data; nothing is fabricated here.

#![forbid(unsafe_code)]

mod complex_matrix;

use complex_matrix::CMat;
use shbt_core_math::precision::PrecisionMode;

pub use num_complex::Complex64 as C64;

/// Canonical crate identifier used by the workbench for topology reporting.
pub const CRATE_NAME: &str = "shbt-dielectric-floquet";

/// Residual convergence target for the inverse Floquet iteration:
/// `‖ε^{(k+1)} − ε^{(k)}‖∞ < 1e-14`.
pub const RESIDUAL_TOLERANCE: f64 = 1e-14;

/// Elementary charge [C].
pub const E_CHARGE: f64 = 1.602_176_634e-19;
/// Vacuum permittivity [F/m].
pub const EPSILON_0: f64 = 8.854_187_812_8e-12;
/// Reduced Planck constant [J·s].
pub const HBAR: f64 = 1.054_571_817e-34;
/// Bohr radius [m] — natural unit for the plane-wave Coulomb factor.
pub const BOHR: f64 = 5.291_772_109_03e-11;
/// Hartree [J].
pub const HARTREE: f64 = 4.359_744_722_207_1e-18;

/// One k-space transition `a → b` with its Floquet–Adler–Wiser matrix
/// elements `M_{ℓ,G}^{ab}(k,q)` for `ℓ ∈ ℓ_range` and every `g_vectors` entry.
#[derive(Clone, Debug)]
pub struct Transition {
    /// `w_k (f_ak − f_{b,k+q})`, occupation difference folded into the weight.
    pub weight: f64,
    /// `ε_ak − ε_{b,k+q}` in the energy unit of `photon_energy`/`drive_energy`.
    pub delta_e: f64,
    /// `elements[ℓ − ℓ_range.start][g]` = `M_{ℓ,G}^{ab}(k,q)`.
    pub elements: Vec<Vec<C64>>,
}

/// Floquet response problem for one `(q, ω)` on a `G`-vector set with output
/// harmonics `m, n` and internal Floquet harmonics `p`.
#[derive(Clone, Debug)]
pub struct FloquetModel {
    /// Unit-cell volume `Ω_cell` [m³] (or consistent area/length in 2D/1D).
    pub cell_volume: f64,
    /// Transfer momentum `q` [1/m].
    pub q: [f64; 3],
    /// Reciprocal vectors `G` [1/m].
    pub g_vectors: Vec<[f64; 3]>,
    /// Output harmonic indices `m, n`.
    pub harmonics: Vec<i64>,
    /// Internal Floquet harmonics `p` summed over.
    pub p_range: (i64, i64),
    /// Range of `ℓ` covered by each `Transition::elements`.
    pub ell_range: (i64, i64),
    /// Photon energy `ℏω` [J or consistent unit].
    pub photon_energy: f64,
    /// Drive energy `ℏΩ_D` (0 for the static Adler–Wiser limit).
    pub drive_energy: f64,
    /// Broadening `η` (same unit as `photon_energy`).
    pub eta: f64,
    /// Aggregated k/band transitions.
    pub transitions: Vec<Transition>,
}

/// Assembled dielectric matrix `ε_{GG'}^{mn}` over composite `(G, m)` indices.
#[derive(Clone, Debug)]
pub struct DielectricMatrix {
    /// `(g_index, m)` for each row/column.
    pub index: Vec<(usize, i64)>,
    /// Dense matrix in composite index order.
    pub matrix: CMat,
    /// Coulomb factor `v_G` per G index [in consistent units].
    pub v_g: Vec<f64>,
}

/// Inversion result.
#[derive(Clone, Debug)]
pub struct InverseResponse {
    /// The inverse dielectric matrix `ε⁻¹` in composite index order.
    pub inverse: CMat,
    /// `(g_index, m)` row/column labels.
    pub index: Vec<(usize, i64)>,
    /// `‖A‖₁‖A⁻¹‖₁` estimate.
    pub condition_number: f64,
    /// Precision tier implied by the condition number.
    pub precision: PrecisionMode,
}

/// Convergence report for a refinement series of [`FloquetModel`]s sharing
/// the same `g_vectors` and `harmonics` (refinement acts on `p_range`,
/// k-sampling, or the transition table).
#[derive(Clone, Debug)]
pub struct ConvergenceReport {
    /// `‖ε⁻¹_k − ε⁻¹_{k−1}‖∞` for each successive pair.
    pub residuals: Vec<f64>,
    /// Whether the last residual is below [`RESIDUAL_TOLERANCE`].
    pub converged: bool,
    /// Final inverse response.
    pub response: InverseResponse,
}

/// Bare Coulomb factor `v_G(q) = e²/(ε₀|q+G|²)`; zero at the singular
/// `|q+G| = 0` head element (macroscopic response is handled separately).
pub fn coulomb_factor(q: [f64; 3], g: [f64; 3]) -> f64 {
    let k = [q[0] + g[0], q[1] + g[1], q[2] + g[2]];
    let k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
    if k2 == 0.0 {
        0.0
    } else {
        E_CHARGE * E_CHARGE / (EPSILON_0 * k2)
    }
}

impl FloquetModel {
    /// `Π_{GG'}^{mn}` at indices `(Gi, mi) → (Gj, nj)` within the retained
    /// `p` range. Matrix elements outside `ell_range` are taken as zero.
    fn polarizability(&self, gi: usize, mi: i64, gj: usize, nj: i64) -> C64 {
        let mut acc = C64::new(0.0, 0.0);
        for tr in &self.transitions {
            let mut inner = C64::new(0.0, 0.0);
            for p in self.p_range.0..=self.p_range.1 {
                let li = mi - p;
                let lj = nj - p;
                let row_i = li - self.ell_range.0;
                let row_j = lj - self.ell_range.0;
                let out_i = row_i < 0 || row_i > self.ell_range.1 - self.ell_range.0;
                let out_j = row_j < 0 || row_j > self.ell_range.1 - self.ell_range.0;
                if out_i || out_j {
                    continue;
                }
                let mi_g = tr.elements[row_i as usize][gi];
                let mj_g = tr.elements[row_j as usize][gj];
                let denom = tr.delta_e
                    + self.photon_energy
                    + p as f64 * self.drive_energy
                    + C64::new(0.0, self.eta);
                inner += mi_g * mj_g.conj() / denom;
            }
            acc += inner * tr.weight;
        }
        acc / self.cell_volume
    }

    /// Assembles `ε_{GG'}^{mn} = δδ − v_G Π_{GG'}^{mn}` in composite order.
    pub fn assemble(&self) -> DielectricMatrix {
        let ng = self.g_vectors.len();
        let index: Vec<(usize, i64)> = (0..ng)
            .flat_map(|g| self.harmonics.iter().map(move |&m| (g, m)))
            .collect();
        let n = index.len();
        let mut matrix = CMat::zeros(n, n);
        let v_g: Vec<f64> = self
            .g_vectors
            .iter()
            .map(|&g| coulomb_factor(self.q, g))
            .collect();
        for (row, &(gi, mi)) in index.iter().enumerate() {
            for (col, &(gj, nj)) in index.iter().enumerate() {
                let delta = if row == col {
                    C64::new(1.0, 0.0)
                } else {
                    C64::ZERO
                };
                matrix[(row, col)] = delta - v_g[gi] * self.polarizability(gi, mi, gj, nj);
            }
        }
        DielectricMatrix { index, matrix, v_g }
    }

    /// Assembles and inverts the dielectric matrix, returning the response
    /// with a `‖·‖₁`-norm condition estimate and the precision tier the
    /// inversion implies (`PrecisionMode::arbitrary_for_condition_number`).
    pub fn response(&self) -> InverseResponse {
        let dm = self.assemble();
        let (inverse, condition_number) = dm.matrix.inverse_with_condition();
        InverseResponse {
            inverse,
            index: dm.index,
            condition_number,
            precision: PrecisionMode::arbitrary_for_condition_number(condition_number),
        }
    }
}

/// Runs a refinement series of models and reports the inverse-response
/// residual series; convergence is `residual < RESIDUAL_TOLERANCE`.
///
/// Every model must share the same `g_vectors`/`harmonics` (same composite
/// index set); refinement acts on `p_range`, `ell_range`, k-sampling or the
/// transition table.
///
/// # Panics
/// If `models` is empty or the composite index sets differ.
pub fn converge_unit_cell(models: &[FloquetModel]) -> ConvergenceReport {
    assert!(!models.is_empty(), "refinement series is empty");
    let mut residuals = Vec::new();
    let mut prev: Option<InverseResponse> = None;
    for model in models {
        let r = model.response();
        if let Some(p) = &prev {
            assert_eq!(
                p.index, r.index,
                "composite index sets differ across series"
            );
            residuals.push((&r.inverse - &p.inverse).max_abs());
        }
        prev = Some(r);
    }
    let response = prev.expect("nonempty");
    ConvergenceReport {
        converged: residuals.last().is_some_and(|&r| r < RESIDUAL_TOLERANCE),
        residuals,
        response,
    }
}

/// Selects the arithmetic tier for inverting a Floquet matrix with the given
/// estimated condition number.
pub fn inversion_precision(condition_number: f64) -> PrecisionMode {
    PrecisionMode::arbitrary_for_condition_number(condition_number)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(p_max: i64, decay: f64) -> FloquetModel {
        // One transition, G = {0}, m = n ∈ {−1,0,1}, M_{ℓ} = decay^|ℓ|.
        let ell_range = (-1 - p_max, 1 + p_max);
        let elements: Vec<Vec<C64>> = (ell_range.0..=ell_range.1)
            .map(|l| vec![C64::new(decay.powi(l.unsigned_abs() as i32), 0.0)])
            .collect();
        FloquetModel {
            cell_volume: 1.0,
            q: [0.0, 0.0, 0.05 / BOHR],
            g_vectors: vec![[0.0, 0.0, 0.0]],
            harmonics: vec![-1, 0, 1],
            p_range: (-p_max, p_max),
            ell_range,
            photon_energy: 0.5 * HARTREE,
            drive_energy: 0.1 * HARTREE,
            eta: 0.01 * HARTREE,
            transitions: vec![Transition {
                weight: 1.0,
                delta_e: 1.0 * HARTREE,
                elements,
            }],
        }
    }

    #[test]
    fn residual_tolerance_is_1e_minus_14() {
        assert_eq!(RESIDUAL_TOLERANCE, 1e-14);
    }

    #[test]
    fn static_limit_matches_scalar_formula() {
        // p restricted to {0}, single G, single harmonic m=0.
        let mut m = model(0, 0.9);
        m.harmonics = vec![0];
        m.p_range = (0, 0);
        m.ell_range = (0, 0);
        m.transitions[0].elements = vec![vec![C64::new(1.0, 0.0)]];
        let dm = m.assemble();
        let v = coulomb_factor(m.q, [0.0; 3]);
        let pi = 1.0 / (HARTREE + 0.5 * HARTREE + C64::new(0.0, 0.01 * HARTREE));
        let expect = C64::new(1.0, 0.0) - v * pi;
        assert!((dm.matrix[(0, 0)] - expect).norm() / expect.norm() < 1e-12);
        let r = m.response();
        assert!(r.condition_number >= 1.0);
    }

    #[test]
    fn identity_when_no_transitions() {
        let mut m = model(2, 0.5);
        m.transitions.clear();
        let dm = m.assemble();
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert_eq!(dm.matrix[(i, j)], C64::new(want, 0.0));
            }
        }
    }

    #[test]
    fn unit_cell_series_converges_below_1e_minus_14() {
        // M_ℓ decays as 0.5^|ℓ|, so widening the p sum converges rapidly.
        let series: Vec<FloquetModel> = (2..=14).map(|p| model(p, 0.5)).collect();
        let report = converge_unit_cell(&series);
        for (i, &r) in report.residuals.iter().enumerate() {
            if i > 0 {
                assert!(
                    r <= report.residuals[i - 1] * 0.9 + 1e-15,
                    "residuals: {:?}",
                    report.residuals
                );
            }
        }
        assert!(report.converged, "residuals: {:?}", report.residuals);
        assert!(*report.residuals.last().unwrap() < RESIDUAL_TOLERANCE);
    }

    #[test]
    fn multi_g_coupling_is_consistent() {
        // Two G vectors; verify the m→m, G→G′ block structure of the formula.
        let mut m = model(0, 0.9);
        m.g_vectors = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.25 / BOHR]];
        m.harmonics = vec![0];
        m.p_range = (0, 0);
        m.ell_range = (0, 0);
        m.transitions[0].elements = vec![vec![C64::new(1.0, 0.0), C64::new(0.3, 0.1)]];
        let dm = m.assemble();
        let denom = HARTREE + 0.5 * HARTREE + C64::new(0.0, 0.01 * HARTREE);
        let (v0, v1) = (dm.v_g[0], dm.v_g[1]);
        // ε_{00} = 1 − v0·1·1*/Δ;  ε_{01} = −v0·M0·M1*/Δ;  ε_{11} = 1 − v1·|M1|²/Δ.
        let e00 = C64::new(1.0, 0.0) - v0 / denom;
        let e01 = -v0 * C64::new(0.3, -0.1) / denom;
        let e11 = C64::new(1.0, 0.0) - v1 * C64::new(0.1, 0.0) / denom;
        assert!((dm.matrix[(0, 0)] - e00).norm() < 1e-30 * e00.norm().max(1.0));
        assert!((dm.matrix[(0, 1)] - e01).norm() < 1e-30 + 1e-3 * e01.norm());
        assert!((dm.matrix[(1, 1)] - e11).norm() < 1e-3 * e11.norm());
    }

    #[test]
    fn well_conditioned_uses_minimum_width() {
        assert_eq!(inversion_precision(10.0).significant_bits(), 128);
    }
}
