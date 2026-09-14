//! DIN 2092 conical disc (Belleville) spring mechanics and DIN 2093 series
//! classification: single-disc force/stiffness/stress (Almen–László), Group 3
//! contact-flat correction, and compound stacks with DIN friction factors.
//!
//! Units are SI throughout (m, N, Pa, K).

use crate::material::{Elastic, LinearModulus};

/// DIN 2092 axial force for one disc from SI geometry and material inputs.
pub fn force_per_disc(
    de: f64,
    di: f64,
    t: f64,
    h0: f64,
    deflection: f64,
    youngs: f64,
    poisson: f64,
) -> f64 {
    DiscSpring {
        outer_diameter: de,
        inner_diameter: di,
        thickness: t,
        cone_height: h0,
    }
    .force(deflection, Elastic { youngs, poisson })
}

/// Conical disc geometry per DIN 2093 (dimensions without contact flats).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiscSpring {
    /// Outer diameter `De` [m].
    pub outer_diameter: f64,
    /// Inner diameter `Di` [m].
    pub inner_diameter: f64,
    /// Material thickness `t` [m].
    pub thickness: f64,
    /// Free cone height `h₀ = l₀ − t` [m].
    pub cone_height: f64,
}

/// DIN 2093 manufacturing group, selected by thickness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Din2093Group {
    /// `t < 1.25 mm`, cold formed, no contact flats.
    Group1,
    /// `1.25 mm ≤ t ≤ 6 mm`, fine blanked / machined edges, no contact flats.
    Group2,
    /// `t > 6 mm`, machined with contact flats and reduced thickness `t'`.
    Group3,
}

/// Loading direction for friction-corrected stack forces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadDirection {
    /// Deflection increasing; friction adds to the external force.
    Loading,
    /// Deflection decreasing; friction subtracts from the external force.
    Unloading,
}

/// DIN 2092 stress components at the characteristic cross-section points.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiscStresses {
    /// Compressive stress at the centre of rotation `σ_OM` [Pa].
    pub sigma_om: f64,
    /// Upper inner edge `σ_I` [Pa].
    pub sigma_i: f64,
    /// Lower inner edge `σ_II` [Pa].
    pub sigma_ii: f64,
    /// Upper outer edge `σ_III` [Pa].
    pub sigma_iii: f64,
    /// Lower outer edge `σ_IV` [Pa].
    pub sigma_iv: f64,
}

impl DiscSpring {
    /// Builds a disc from millimetre catalogue dimensions `De, Di, t, l₀`.
    pub fn from_mm(de: f64, di: f64, t: f64, l0: f64) -> Self {
        Self {
            outer_diameter: de * 1e-3,
            inner_diameter: di * 1e-3,
            thickness: t * 1e-3,
            cone_height: (l0 - t) * 1e-3,
        }
    }

    /// Diameter ratio `δ = De / Di`.
    pub fn delta(&self) -> f64 {
        self.outer_diameter / self.inner_diameter
    }

    /// `h₀ / t`, the characteristic-curve shape parameter.
    pub fn height_ratio(&self) -> f64 {
        self.cone_height / self.thickness
    }

    /// DIN 2093 group classification by thickness.
    pub fn group(&self) -> Din2093Group {
        let t_mm = self.thickness * 1e3;
        if t_mm < 1.25 {
            Din2093Group::Group1
        } else if t_mm <= 6.0 {
            Din2093Group::Group2
        } else {
            Din2093Group::Group3
        }
    }

    /// Geometry factor `K₁` (DIN 2092; cf.pdf Eq. 173 `M`).
    pub fn k1(&self) -> f64 {
        let d = self.delta();
        let ln = d.ln();
        ((d - 1.0) / d).powi(2) / (std::f64::consts::PI * ((d + 1.0) / (d - 1.0) - 2.0 / ln))
    }

    /// Stress factor `K₂`.
    pub fn k2(&self) -> f64 {
        let d = self.delta();
        let ln = d.ln();
        (6.0 / std::f64::consts::PI) * ((d - 1.0) / ln - 1.0) / ln
    }

    /// Stress factor `K₃`.
    pub fn k3(&self) -> f64 {
        let d = self.delta();
        (3.0 / std::f64::consts::PI) * (d - 1.0) / d.ln()
    }

    /// Neutral (rotation-centre) radius `r₀ = (De − Di) / (2 ln δ)`.
    pub fn neutral_radius(&self) -> f64 {
        (self.outer_diameter - self.inner_diameter) / (2.0 * self.delta().ln())
    }

    /// Common prefactor `4E/(1−ν²) · t⁴/(K₁ De²)` [N].
    fn force_scale(&self, e: Elastic) -> f64 {
        4.0 * e.plate_modulus() * self.thickness.powi(4)
            / (self.k1() * self.outer_diameter * self.outer_diameter)
    }

    /// DIN 2092 axial force at deflection `s` for the ideal disc without
    /// contact flats (cf.pdf Eq. 172):
    /// `F = 4E/(1−ν²) · t⁴/(K₁De²) · (s/t) [ (h₀/t − s/t)(h₀/t − s/2t) + 1 ]`.
    pub fn force(&self, s: f64, e: Elastic) -> f64 {
        let x = s / self.thickness;
        let h = self.height_ratio();
        self.force_scale(e) * x * ((h - x) * (h - 0.5 * x) + 1.0)
    }

    /// Tangent stiffness `dF/ds` [N/m] (cf.pdf Eq. 172, second line).
    pub fn stiffness(&self, s: f64, e: Elastic) -> f64 {
        let x = s / self.thickness;
        let h = self.height_ratio();
        self.force_scale(e) / self.thickness * (h * h - 3.0 * h * x + 1.5 * x * x + 1.0)
    }

    /// Force when pressed flat (`s = h₀`).
    pub fn flat_force(&self, e: Elastic) -> f64 {
        self.force(self.cone_height, e)
    }

    /// Group 3 force with contact flats and reduced thickness `t'`
    /// (DIN 2092 `K₄` correction). `reduced_thickness` is typically `0.94 t`.
    pub fn force_with_contact_flats(&self, s: f64, reduced_thickness: f64, e: Elastic) -> f64 {
        let t = self.thickness;
        let tp = reduced_thickness;
        let l0 = self.cone_height + t;
        let h0p = l0 - tp;
        let r = tp / t;
        let c1 = (r * r) / (0.25 * (l0 / t) - r + 0.75);
        let c2 = c1 / (r * r * r) * (5.0 / 32.0 * (l0 / t - 1.0).powi(2) + 1.0);
        let k4 = (-0.5 * c1 + (0.25 * c1 * c1 + c2).sqrt()).sqrt();
        let scale = 4.0 * e.plate_modulus() * tp.powi(4)
            / (self.k1() * self.outer_diameter * self.outer_diameter);
        let x = s / tp;
        let h = h0p / tp;
        scale * k4 * k4 * x * (k4 * k4 * (h - x) * (h - 0.5 * x) + 1.0)
    }

    /// DIN 2092 characteristic stresses at deflection `s` (negative = compression).
    pub fn stresses(&self, s: f64, e: Elastic) -> DiscStresses {
        let t = self.thickness;
        let x = s / t;
        let h = self.height_ratio();
        let base = -4.0 * e.plate_modulus() * t * t
            / (self.k1() * self.outer_diameter * self.outer_diameter)
            * x;
        let (k2, k3, d) = (self.k2(), self.k3(), self.delta());
        let m = h - 0.5 * x;
        DiscStresses {
            sigma_om: base * 3.0 / std::f64::consts::PI,
            sigma_i: base * (k2 * m + k3),
            sigma_ii: base * (k2 * m - k3),
            sigma_iii: base / d * ((k2 - 2.0 * k3) * m - k3),
            sigma_iv: base / d * ((k2 - 2.0 * k3) * m + k3),
        }
    }
}

/// Belleville stack layout: `parallel` washers nested per group share the
/// force, `series` groups share the deflection (DIN 2092 compound stacks).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StackLayout {
    /// Washers nested in parallel per group (`n`).
    pub parallel: u32,
    /// Groups arranged in series (`i`).
    pub series: u32,
}

impl StackLayout {
    /// Total force of the stack given the force of a single washer at the
    /// per-group deflection.
    pub fn stack_force(self, single_washer_force: f64) -> f64 {
        single_washer_force * f64::from(self.parallel)
    }

    /// Total deflection of the stack given a single group's deflection.
    pub fn stack_deflection(self, group_deflection: f64) -> f64 {
        group_deflection * f64::from(self.series)
    }
}

/// DIN 2092 friction coefficients for compound stacks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StackFriction {
    /// Surface friction between nested parallel discs `w_M` (0.005 – 0.03).
    pub parallel: f64,
    /// Edge friction between series groups `w_R` (0.02 – 0.04).
    pub series: f64,
}

impl StackFriction {
    /// Ideal frictionless stack.
    pub const NONE: Self = Self {
        parallel: 0.0,
        series: 0.0,
    };
}

/// A compound stack of identical discs at a shared temperature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiscStack {
    /// Disc geometry.
    pub disc: DiscSpring,
    /// Stack arrangement.
    pub layout: StackLayout,
    /// Friction model.
    pub friction: StackFriction,
    /// Spring alloy modulus law.
    pub modulus: LinearModulus,
}

impl DiscStack {
    /// Per-disc deflection `s = x / i` for stack travel `x`.
    pub fn disc_deflection(&self, stack_travel: f64) -> f64 {
        stack_travel / f64::from(self.layout.series)
    }

    /// Frictionless stack force at travel `x` and temperature `t` [K].
    pub fn ideal_force(&self, stack_travel: f64, t: f64) -> f64 {
        let s = self.disc_deflection(stack_travel);
        self.layout
            .stack_force(self.disc.force(s, self.modulus.at(t)))
    }

    /// DIN 2092 friction-corrected force:
    /// `F = n F₁(s) / (1 ∓ w_M (n − 1) ∓ w_R)`, `−` when loading.
    pub fn force(&self, stack_travel: f64, t: f64, direction: LoadDirection) -> f64 {
        let n = f64::from(self.layout.parallel);
        let corr = self.friction.parallel * (n - 1.0) + self.friction.series;
        let denom = match direction {
            LoadDirection::Loading => 1.0 - corr,
            LoadDirection::Unloading => 1.0 + corr,
        };
        self.ideal_force(stack_travel, t) / denom
    }

    /// Frictionless load-path stiffness `dF/dx = n k₁(s) / i` [N/m].
    pub fn stiffness(&self, stack_travel: f64, t: f64) -> f64 {
        let s = self.disc_deflection(stack_travel);
        f64::from(self.layout.parallel) * self.disc.stiffness(s, self.modulus.at(t))
            / f64::from(self.layout.series)
    }

    /// Stiffness per loaded area `K = (dF/dx)/A` [Pa/m] (cf.pdf Eq. 171).
    pub fn pressure_stiffness(&self, stack_travel: f64, t: f64, loaded_area: f64) -> f64 {
        self.stiffness(stack_travel, t) / loaded_area
    }

    /// Preload pressure change `Δp = (1/A) ∫ k(x, T) dx` over a thermal
    /// travel excursion, integrated with 32-point Gauss–Legendre quadrature.
    pub fn pressure_excursion(&self, x0: f64, dx: f64, t: f64, loaded_area: f64) -> f64 {
        let (nodes, weights) = crate::quadrature::gauss_legendre_32();
        let half = 0.5 * dx;
        let mid = x0 + half;
        let mut acc = 0.0;
        for (xi, w) in nodes.iter().zip(weights) {
            acc += w * self.stiffness(mid + half * xi, t);
        }
        acc * half / loaded_area
    }
}

/// DIN 2093 catalogue entry (Group 2, series A/B) with the tabulated force at
/// the standard test deflection `s = 0.75 h₀`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Din2093Entry {
    /// Catalogue designation, e.g. `"A 40"`.
    pub designation: &'static str,
    /// `De` [mm].
    pub de_mm: f64,
    /// `Di` [mm].
    pub di_mm: f64,
    /// `t` [mm].
    pub t_mm: f64,
    /// Free height `l₀` [mm].
    pub l0_mm: f64,
    /// Tabulated force at `s = 0.75 h₀` [N].
    pub force_075_n: f64,
}

impl Din2093Entry {
    /// Geometry of the catalogue disc.
    pub fn disc(&self) -> DiscSpring {
        DiscSpring::from_mm(self.de_mm, self.di_mm, self.t_mm, self.l0_mm)
    }
}

macro_rules! entry {
    ($name:literal, $de:expr, $di:expr, $t:expr, $l0:expr, $f:expr) => {
        Din2093Entry {
            designation: $name,
            de_mm: $de,
            di_mm: $di,
            t_mm: $t,
            l0_mm: $l0,
            force_075_n: $f,
        }
    };
}

/// DIN 2093 Group 2 (1.25 mm ≤ t ≤ 6 mm) series A (`h₀/t ≈ 0.4`) and series B
/// (`h₀/t ≈ 0.75`) reference discs with the standard force at `s = 0.75 h₀`
/// (`E = 206 GPa`, `ν = 0.3`, three significant figures as tabulated).
pub const DIN_2093_GROUP2: [Din2093Entry; 20] = [
    entry!("A 31.5", 31.5, 16.3, 1.75, 2.45, 3_870.0),
    entry!("A 40", 40.0, 20.4, 2.25, 3.15, 6_500.0),
    entry!("A 45", 45.0, 22.4, 2.5, 3.5, 7_720.0),
    entry!("A 50", 50.0, 25.4, 3.0, 4.1, 12_000.0),
    entry!("A 56", 56.0, 28.5, 3.0, 4.3, 11_400.0),
    entry!("A 63", 63.0, 31.0, 3.5, 4.9, 15_000.0),
    entry!("A 71", 71.0, 36.0, 4.0, 5.6, 20_500.0),
    entry!("A 80", 80.0, 41.0, 5.0, 6.7, 33_600.0),
    entry!("A 90", 90.0, 46.0, 5.0, 7.0, 31_400.0),
    entry!("A 100", 100.0, 51.0, 6.0, 8.2, 48_000.0),
    entry!("B 31.5", 31.5, 16.3, 1.25, 2.15, 1_910.0),
    entry!("B 40", 40.0, 20.4, 1.5, 2.65, 2_620.0),
    entry!("B 45", 45.0, 22.4, 1.75, 3.05, 3_650.0),
    entry!("B 50", 50.0, 25.4, 2.0, 3.4, 4_760.0),
    entry!("B 56", 56.0, 28.5, 2.0, 3.6, 4_440.0),
    entry!("B 63", 63.0, 31.0, 2.5, 4.25, 7_190.0),
    entry!("B 71", 71.0, 36.0, 2.5, 4.5, 6_720.0),
    entry!("B 80", 80.0, 41.0, 3.0, 5.3, 10_500.0),
    entry!("B 90", 90.0, 46.0, 3.5, 6.0, 14_200.0),
    entry!("B 100", 100.0, 51.0, 3.5, 6.3, 13_100.0),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::{T_COLD, T_HOT};

    #[test]
    fn k1_matches_cf_eq_173() {
        let disc = DiscSpring {
            outer_diameter: 0.250,
            inner_diameter: 0.125,
            thickness: 12.0e-3,
            cone_height: 17.4e-3,
        };
        assert!((disc.k1() - 0.694_333_20).abs() < 1e-8, "{}", disc.k1());
        assert_eq!(disc.group(), Din2093Group::Group3);
    }

    #[test]
    fn din2093_group2_force_deflection_within_half_percent() {
        let e = LinearModulus::DIN_2093_STEEL.at(293.15);
        let mut worst = 0.0f64;
        for entry in DIN_2093_GROUP2 {
            let disc = entry.disc();
            assert_eq!(disc.group(), Din2093Group::Group2, "{}", entry.designation);
            let f = disc.force(0.75 * disc.cone_height, e);
            let rel = (f / entry.force_075_n - 1.0).abs();
            worst = worst.max(rel);
            assert!(
                rel < 5e-3,
                "{}: model {f:.1} N vs table {} N ({:.3}%)",
                entry.designation,
                entry.force_075_n,
                rel * 100.0
            );
        }
        assert!(worst > 0.0);
    }

    #[test]
    fn stiffness_is_force_derivative() {
        let disc = DIN_2093_GROUP2[1].disc();
        let e = LinearModulus::DIN_2093_STEEL.at(293.15);
        for frac in [0.1, 0.5, 0.9] {
            let s = frac * disc.cone_height;
            let h = 1e-9;
            let fd = (disc.force(s + h, e) - disc.force(s - h, e)) / (2.0 * h);
            assert!((fd / disc.stiffness(s, e) - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn contact_flats_soften_group3_disc() {
        let disc = DiscSpring::from_mm(125.0, 64.0, 8.0, 10.6);
        assert_eq!(disc.group(), Din2093Group::Group3);
        let e = LinearModulus::DIN_2093_STEEL.at(293.15);
        let s = 0.5 * disc.cone_height;
        let ideal = disc.force(s, e);
        let reduced = disc.force_with_contact_flats(s, 0.94 * disc.thickness, e);
        assert!(
            reduced < ideal && reduced > 0.5 * ideal,
            "{reduced} vs {ideal}"
        );
    }

    #[test]
    fn cf_four_disc_stack_pressure_stiffness() {
        let stack = DiscStack {
            disc: DiscSpring {
                outer_diameter: 0.250,
                inner_diameter: 0.125,
                thickness: 12.0e-3,
                cone_height: 17.4e-3,
            },
            layout: StackLayout {
                parallel: 1,
                series: 4,
            },
            friction: StackFriction::NONE,
            modulus: LinearModulus::INCONEL_X750,
        };
        let x0 = 18.25e-3;
        assert!((stack.disc_deflection(x0) - 4.5625e-3).abs() < 1e-15);
        let k_cold = stack.stiffness(x0, T_COLD);
        let k_hot = stack.stiffness(x0, T_HOT);
        assert!(k_hot < k_cold);
        assert!((k_hot / k_cold - 175.437 / 193.0).abs() < 1e-3);
        let excursion = stack.pressure_excursion(x0, 7.7952e-6, T_HOT, 1.0);
        let linear = k_hot * 7.7952e-6;
        assert!((excursion / linear - 1.0).abs() < 1e-3);
    }

    #[test]
    fn friction_brackets_ideal_force() {
        let stack = DiscStack {
            disc: DIN_2093_GROUP2[1].disc(),
            layout: StackLayout {
                parallel: 2,
                series: 3,
            },
            friction: StackFriction {
                parallel: 0.02,
                series: 0.03,
            },
            modulus: LinearModulus::DIN_2093_STEEL,
        };
        let x = 3.0 * 0.5e-3;
        let ideal = stack.ideal_force(x, 293.15);
        let up = stack.force(x, 293.15, LoadDirection::Loading);
        let down = stack.force(x, 293.15, LoadDirection::Unloading);
        assert!(down < ideal && ideal < up);
        assert!((up / ideal - 1.0 / 0.95).abs() < 1e-12);
    }

    #[test]
    fn stresses_have_din_signs() {
        let disc = DIN_2093_GROUP2[1].disc();
        let e = LinearModulus::DIN_2093_STEEL.at(293.15);
        let s = disc.stresses(0.75 * disc.cone_height, e);
        assert!(s.sigma_om < 0.0 && s.sigma_i < 0.0);
        assert!(s.sigma_iii > 0.0, "{:?}", s);
        assert!(s.sigma_i.abs() > s.sigma_om.abs());
    }
}
