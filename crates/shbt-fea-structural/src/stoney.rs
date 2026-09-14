//! Extended Stoney thin-film stress with temperature-dependent thermal
//! mismatch and pre/post-deposition curvature radii (simulator_spec.pdf §4,
//! "HiPIMS Layer Deposition & Film Stress Diagnostics").
//!
//! `σ_f = M_s t_s² / (6 t_f) · (1/R_post − 1/R_pre) · C(t_f/t_s, M_f/M_s)`
//!
//! where `M = E/(1−ν)` is the biaxial modulus and `C` is the finite-thickness
//! correction that reduces to 1 in the Stoney limit `t_f ≪ t_s`. The thermal
//! contribution over a temperature excursion is
//! `σ_th(T) = −M_f ∫_{T_dep}^{T} [α_f(T') − α_s(T')] dT'` (compressive when the
//! film expands more than the substrate), and the substrate strain that
//! distorts the grating pitch is the sum of the free thermal expansion and the
//! curvature-induced surface strain.

use crate::material::{Elastic, LinearCte};

/// One layer of the film/substrate bilayer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Layer {
    /// Elastic constants.
    pub elastic: Elastic,
    /// Thickness [m].
    pub thickness: f64,
    /// Thermal expansion law.
    pub cte: LinearCte,
}

/// Curvature radii measured by deflectometry [m]; `f64::INFINITY` for flat.
/// Positive radius: centre of curvature on the film side (tensile film).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurvatureRadii {
    /// Substrate radius before deposition.
    pub r_pre: f64,
    /// Substrate radius after deposition.
    pub r_post: f64,
}

impl CurvatureRadii {
    /// Curvature change `Δκ = 1/R_post − 1/R_pre`.
    pub fn delta_curvature(self) -> f64 {
        1.0 / self.r_post - 1.0 / self.r_pre
    }
}

/// Film/substrate system.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilmOnSubstrate {
    /// Deposited film.
    pub film: Layer,
    /// Substrate (e.g. CVD diamond carrying the Option B grating).
    pub substrate: Layer,
}

/// Decomposed film stress state [Pa].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilmStress {
    /// Stress inferred from curvature at the measurement temperature.
    pub intrinsic: f64,
    /// Thermal-mismatch stress accumulated between `t_measure` and `t`.
    pub thermal: f64,
    /// `intrinsic + thermal`.
    pub total: f64,
}

impl FilmOnSubstrate {
    /// Finite-thickness correction factor `C = (1 + m h³) / (1 + h)` with
    /// `h = t_f/t_s`, `m = M_f/M_s`, obtained from the exact elastic bilayer
    /// (force and moment balance with a uniform mismatch strain). Equals 1 as
    /// `h → 0`.
    pub fn thickness_correction(&self) -> f64 {
        let h = self.film.thickness / self.substrate.thickness;
        let m = self.film.elastic.biaxial_modulus() / self.substrate.elastic.biaxial_modulus();
        (1.0 + m * h * h * h) / (1.0 + h)
    }

    /// Classical Stoney stress `M_s t_s² Δκ / (6 t_f)`.
    pub fn stoney_stress(&self, radii: CurvatureRadii) -> f64 {
        self.substrate.elastic.biaxial_modulus()
            * self.substrate.thickness.powi(2)
            * radii.delta_curvature()
            / (6.0 * self.film.thickness)
    }

    /// Extended Stoney stress with finite-thickness correction.
    pub fn extended_stoney_stress(&self, radii: CurvatureRadii) -> f64 {
        self.stoney_stress(radii) * self.thickness_correction()
    }

    /// Curvature produced by a uniform biaxial mismatch strain `ε_m` (film
    /// stress-free strain relative to the substrate; exact bilayer plate
    /// solution). A film that wants to expand (`ε_m > 0`) ends up compressive
    /// and bends the substrate away from the film, i.e. `κ < 0`.
    pub fn curvature_from_mismatch(&self, mismatch_strain: f64) -> f64 {
        let (tf, ts) = (self.film.thickness, self.substrate.thickness);
        let (mf, ms) = (
            self.film.elastic.biaxial_modulus(),
            self.substrate.elastic.biaxial_modulus(),
        );
        let num = 6.0 * mf * ms * tf * ts * (tf + ts) * mismatch_strain;
        let den = mf * mf * tf.powi(4)
            + 4.0 * mf * ms * tf.powi(3) * ts
            + 6.0 * mf * ms * tf * tf * ts * ts
            + 4.0 * mf * ms * tf * ts.powi(3)
            + ms * ms * ts.powi(4);
        -num / den
    }

    /// Thermal mismatch strain `∫ (α_f − α_s) dT` from `t0` to `t1`.
    pub fn thermal_mismatch_strain(&self, t0: f64, t1: f64) -> f64 {
        self.film.cte.strain(t0, t1) - self.substrate.cte.strain(t0, t1)
    }

    /// Full stress decomposition: curvature-derived intrinsic stress at
    /// `t_measure`, plus thermal stress on heating/cooling to `t`.
    pub fn film_stress(&self, radii: CurvatureRadii, t_measure: f64, t: f64) -> FilmStress {
        let intrinsic = self.extended_stoney_stress(radii);
        let thermal =
            -self.film.elastic.biaxial_modulus() * self.thermal_mismatch_strain(t_measure, t);
        FilmStress {
            intrinsic,
            thermal,
            total: intrinsic + thermal,
        }
    }

    /// Predicted post-deposition curvature at temperature `t` given the
    /// measured radii at `t_measure`.
    pub fn curvature_at(&self, radii: CurvatureRadii, t_measure: f64, t: f64) -> f64 {
        1.0 / radii.r_post
            + self.curvature_from_mismatch(self.thermal_mismatch_strain(t_measure, t))
    }

    /// In-plane strain of the substrate top surface (grating plane) relative
    /// to the cold reference: free thermal expansion plus curvature bending
    /// strain `κ t_s / 2` at the film-side surface.
    pub fn substrate_surface_strain(&self, radii: CurvatureRadii, t_ref: f64, t: f64) -> f64 {
        let free = self.substrate.cte.strain(t_ref, t);
        let kappa = self.curvature_at(radii, t_ref, t);
        free + 0.5 * kappa * self.substrate.thickness
    }

    /// Distorted grating pitch `Λ(T) = Λ₀ (1 + ε_surface)`.
    pub fn distorted_pitch(
        &self,
        nominal_pitch: f64,
        radii: CurvatureRadii,
        t_ref: f64,
        t: f64,
    ) -> f64 {
        nominal_pitch * (1.0 + self.substrate_surface_strain(radii, t_ref, t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::{T_COLD, T_HOT};

    fn system(tf: f64) -> FilmOnSubstrate {
        FilmOnSubstrate {
            film: Layer {
                elastic: Elastic {
                    youngs: 150.0e9,
                    poisson: 0.38,
                },
                thickness: tf,
                cte: LinearCte {
                    a: 11.8e-6,
                    b: 0.0,
                    t_ref: T_COLD,
                },
            },
            substrate: Layer {
                elastic: Elastic {
                    youngs: 1050.0e9,
                    poisson: 0.10,
                },
                thickness: 500.0e-6,
                cte: LinearCte::CVD_DIAMOND,
            },
        }
    }

    #[test]
    fn correction_tends_to_one_for_thin_films() {
        let c = system(50.0e-9).thickness_correction();
        assert!((c - 1.0).abs() < 1e-3, "{c}");
        let thick = system(50.0e-6).thickness_correction();
        assert!((thick - 1.0).abs() > 0.05 && thick < 1.0, "{thick}");
    }

    #[test]
    fn curvature_and_stoney_are_mutually_consistent() {
        let sys = system(50.0e-9);
        let eps = 1e-3;
        let kappa = sys.curvature_from_mismatch(eps);
        let radii = CurvatureRadii {
            r_pre: f64::INFINITY,
            r_post: 1.0 / kappa,
        };
        let sigma = sys.extended_stoney_stress(radii);
        let expected = -sys.film.elastic.biaxial_modulus() * eps;
        assert!(
            (sigma / expected - 1.0).abs() < 2e-3,
            "{sigma} vs {expected}"
        );
    }

    #[test]
    fn heating_pd_on_diamond_is_compressive_and_grows_pitch() {
        let sys = system(50.0e-9);
        let radii = CurvatureRadii {
            r_pre: 40.0,
            r_post: 25.0,
        };
        let s = sys.film_stress(radii, T_COLD, T_HOT);
        assert!(s.thermal < 0.0);
        assert!((s.total - (s.intrinsic + s.thermal)).abs() < 1.0);
        let pitch = sys.distorted_pitch(960.80e-9, radii, T_COLD, T_HOT);
        let free = 960.80e-9 * (1.0 + 1.0e-6 * 325.0);
        assert!(pitch > 960.80e-9);
        assert!((pitch / free - 1.0).abs() < 1e-3, "{pitch} vs {free}");
    }
}
