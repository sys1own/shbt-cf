//! Option B grating reactor model (cf.pdf §"Pump Light Absorption"):
//! 960.80 nm lamellar Pd–Ir relief, 42.50 nm deep, 7.50 nm backplane,
//! 10 nm Ti adhesion layer, on the CVD diamond window, excited from the
//! fused-silica prism at a 65° internal angle by the dual pump (785.0 /
//! 802.5 nm).

use crate::cmat::c64;
use crate::grating::{Profile, Segment, Truncation};
use crate::material::PumpConstants;
use crate::material::RefractiveIndex;
use crate::solver::{Excitation, Lattice, Layer, Scattering, Side, Stack};

/// Relief period `Λ₀` [m].
pub const PITCH: f64 = 960.80e-9;
/// Relief depth `d` [m].
pub const RELIEF_DEPTH: f64 = 42.50e-9;
/// Mean relief/backplane thickness `t_b` [m].
pub const BACKPLANE: f64 = 7.50e-9;
/// Duty cycle `f_g` (metal ridge fill).
pub const DUTY_CYCLE: f64 = 0.50;
/// Ti adhesion layer thickness [m].
pub const TI_THICKNESS: f64 = 10.0e-9;
/// Internal silica incidence angle [rad].
pub const INCIDENCE_ANGLE: f64 = 65.0 * std::f64::consts::PI / 180.0;
/// Nominal harmonic truncation `m ∈ [−12, 12]` (cf.pdf §"Option B").
pub const TRUNCATION: Truncation = Truncation { m: 12, n: 0 };

/// `Stack` for the Option B grating at one pump wavelength (invariant-y
/// lamellar relief; the grooves are dielectric gaps filled with silica).
pub fn option_b_stack(pump: PumpConstants) -> Stack {
    let eps_si = pump.silica.permittivity();
    let eps_pd = pump.pd_ir.permittivity();
    Stack {
        superstrate: eps_si,
        layers: vec![
            Layer {
                thickness: RELIEF_DEPTH,
                profile: Profile::Lamellar {
                    background: eps_si,
                    segments: vec![Segment {
                        center: 0.5,
                        width: DUTY_CYCLE,
                        eps: eps_pd,
                    }],
                },
            },
            Layer {
                thickness: BACKPLANE,
                profile: Profile::Uniform(eps_pd),
            },
            Layer {
                thickness: TI_THICKNESS,
                profile: Profile::Uniform(pump.ti.permittivity()),
            },
        ],
        substrate: pump.diamond.permittivity(),
        lattice: Lattice {
            period_x: PITCH,
            period_y: None,
        },
    }
}

/// TM-polarised pump excitation at the 65° internal angle.
pub fn option_b_excitation(pump: PumpConstants) -> Excitation {
    Excitation::tm(pump.wavelength, INCIDENCE_ANGLE)
}

/// Scattering + field-enhancement result for one pump wavelength.
#[derive(Clone, Debug)]
pub struct PumpResult {
    /// Full scattering result (order efficiencies, S-matrix).
    pub scattering: Scattering,
    /// Specular reflectance `R₀₀`.
    pub specular_reflectance: f64,
    /// `F_z = |E_z/E₀|²` on the diamond side of the Ti/diamond interface,
    /// maximised over `x` (the grating-assisted comparison observable).
    pub fz_ti_diamond: f64,
    /// `|E_z/E₀|²` on the silica side of the prism/relief interface, maximised
    /// over `x` (metal-surface sampling location; carries the tooth-edge
    /// singularity, so it grows slowly with harmonic count).
    pub fz_metal_surface: f64,
    /// Peak `|E/E₀|²` on the silica side of the prism/relief interface.
    pub total_enhancement: f64,
    /// Position `x/Λ` of the metal-surface `|E_z|` peak.
    pub peak_position: f64,
}

/// Runs the Option B model at `pump` and reports specular reflectance plus the
/// normal-field enhancement `F_z = |E_z(z⁺_{Ti/diamond})/E₀|²` sampled on the
/// dielectric (diamond) side of the Ti/diamond interface, maximised over `x`
/// across one period (cf.pdf Eq. 130 and §"grating-assisted" convention).
pub fn run_pump(pump: PumpConstants, tr: Truncation, x_samples: usize) -> PumpResult {
    let stack = option_b_stack(pump);
    let exc = option_b_excitation(pump);
    let sc = stack.solve(&exc, tr);
    let bottom = stack.layers.len();
    let mut fz_td = 0.0f64;
    let mut fz_ms = 0.0f64;
    let mut e_max = 0.0f64;
    let mut peak = 0.0f64;
    for i in 0..x_samples {
        let x = PITCH * i as f64 / x_samples as f64;
        fz_td = fz_td.max(sc.interface_field(bottom, Side::Below, x, 0.0).z.norm_sqr());
        let f = sc.interface_field(0, Side::Above, x, 0.0);
        let ez = f.z.norm_sqr();
        if ez > fz_ms {
            fz_ms = ez;
            peak = i as f64 / x_samples as f64;
        }
        e_max = e_max.max(f.intensity());
    }
    PumpResult {
        specular_reflectance: sc.specular_reflectance(),
        fz_ti_diamond: fz_td,
        fz_metal_surface: fz_ms,
        total_enhancement: e_max,
        peak_position: peak,
        scattering: sc,
    }
}

/// Planar four-layer comparison of cf.pdf Eq. (194–195) / Tables XXI–XXII:
/// SF11 prism / 1.0 nm Pd–Ir / 10.0 nm Ti / semi-infinite CVD diamond.
/// Uses the Table XXI comparison constants (distinct from Table VIII).
pub fn planar_stack(pump: PumpConstants) -> Stack {
    let _ = pump;
    Stack {
        superstrate: RefractiveIndex::lossless(1.76552).permittivity(),
        layers: vec![
            Layer {
                thickness: 1.0e-9,
                profile: Profile::Uniform(RefractiveIndex { n: 1.835, k: 4.535 }.permittivity()),
            },
            Layer {
                thickness: TI_THICKNESS,
                profile: Profile::Uniform(RefractiveIndex { n: 2.800, k: 3.800 }.permittivity()),
            },
        ],
        substrate: RefractiveIndex::lossless(2.417).permittivity(),
        lattice: Lattice {
            period_x: PITCH,
            period_y: None,
        },
    }
}

/// `|E_z/E₀|²` on the diamond side of the Ti/diamond interface at a given
/// internal incidence angle [rad] for the planar comparison stack.
pub fn planar_ez_enhancement_at(pump: PumpConstants, theta: f64, tr: Truncation) -> f64 {
    let stack = planar_stack(pump);
    let sc = stack.solve(&Excitation::tm(pump.wavelength, theta), tr);
    sc.interface_field(stack.layers.len(), Side::Below, 0.0, 0.0)
        .z
        .norm_sqr()
}

/// `F_z` of the planar comparison at the 65° operating angle.
pub fn planar_ez_enhancement(pump: PumpConstants, tr: Truncation) -> f64 {
    planar_ez_enhancement_at(pump, INCIDENCE_ANGLE, tr)
}

/// Complex specular reflection amplitude `rₚ` for the planar reference.
pub fn planar_reflection_amplitude(pump: PumpConstants, tr: Truncation) -> c64 {
    let sc = planar_stack(pump).solve(&option_b_excitation(pump), tr);
    let a = sc.specular_amplitude();
    // TM specular: E_x = −r_p cos θ.
    -a.x / INCIDENCE_ANGLE.cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::{PUMP_785, PUMP_802_5};

    #[test]
    fn option_b_scattering_is_physical() {
        for pump in [PUMP_785, PUMP_802_5] {
            let r = run_pump(pump, TRUNCATION, 200);
            assert!(
                (0.0..=1.0).contains(&r.specular_reflectance),
                "λ {}: R {}",
                pump.wavelength,
                r.specular_reflectance
            );
            assert!(r.scattering.absorptance() > 0.0);
            assert!(r.fz_metal_surface > 0.0 && r.fz_ti_diamond > 0.0);
            println!(
                "λ={:.1} nm: R00={:.4} A={:.3} |Ez/E0|2={:.1} |E/E0|2={:.1} x={:.2}",
                pump.wavelength * 1e9,
                r.specular_reflectance,
                r.scattering.absorptance(),
                r.fz_metal_surface,
                r.total_enhancement,
                r.peak_position
            );
        }
    }

    #[test]
    fn planar_reference_matches_cf_planar_enhancement() {
        // cf.pdf Table XXII: planar F_z = 0.099945 at 65°, 0.094183 at the
        // 68.516° reflectance minimum.
        let f65 = planar_ez_enhancement(PUMP_785, TRUNCATION);
        assert!((f65 - 0.0999).abs() < 0.02, "{f65}");
        let f_min = planar_ez_enhancement_at(PUMP_785, 68.51617f64.to_radians(), TRUNCATION);
        assert!((f_min - 0.0942).abs() < 0.02, "{f_min}");
    }
}
