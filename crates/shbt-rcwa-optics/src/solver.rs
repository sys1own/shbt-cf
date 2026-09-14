//! Full-vector RCWA with Redheffer star-product S-matrix recursion.
//!
//! Formulation (Moharam–Gaylord–Pommet–Grann 1995 eigenmodes, Li 1996
//! factorisation, Rumpf 2011 gap-medium scattering matrices). All lengths are
//! in metres; wave-vectors are normalised by `k₀ = 2π/λ`, the magnetic field
//! by `H̃ = −i η₀ H`, and the incident electric field has unit amplitude so
//! that squared field magnitudes are enhancement factors `|E/E₀|²` directly.
//!
//! For each layer with convolution matrices `E_xx, E_yy, E_zz` (see
//! [`crate::grating`]) the tangential fields obey `dS/dz = i k₀ P U`,
//! `dU/dz = i k₀ Q S` with
//!
//! ```text
//! P = [ Kx Ezz⁻¹ Ky        I − Kx Ezz⁻¹ Kx ]     Q = [ Kx Ky        Eyy − Kx² ]
//!     [ Ky Ezz⁻¹ Ky − I   −Ky Ezz⁻¹ Kx    ]         [ Ky² − Exx   −Ky Kx    ]
//! ```
//!
//! which for a `y`-invariant TM problem (`Ky = 0`) reduces to cf.pdf Eq. (190):
//! `dSx/dz = i k₀ (I − Kx E⁻¹ Kx) Uy`, `dUy/dz = i k₀ ⟦1/ε⟧⁻¹ Sx`,
//! `Sz = −(1/k₀) E⁻¹ Kx Uy`.

use crate::cmat::{
    block2, c64, diag, diag_re, eig, eye, inv, solve, split_rows, vstack, CMat, ONE, ZERO,
};
use crate::grating::{ConvolutionMatrices, Profile, Truncation};
use faer::prelude::*;

/// One finite-thickness layer.
#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    /// Thickness [m].
    pub thickness: f64,
    /// Unit-cell permittivity distribution.
    pub profile: Profile,
}

/// Lateral periodicity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lattice {
    /// Period along `x` [m].
    pub period_x: f64,
    /// Period along `y` [m]; `None` for `y`-invariant gratings (then only
    /// `n = 0` harmonics are admissible).
    pub period_y: Option<f64>,
}

/// Semi-infinite superstrate / finite layers / semi-infinite substrate.
#[derive(Clone, Debug, PartialEq)]
pub struct Stack {
    /// Incidence half-space permittivity.
    pub superstrate: c64,
    /// Layers from the superstrate downwards.
    pub layers: Vec<Layer>,
    /// Transmission half-space permittivity.
    pub substrate: c64,
    /// Lateral lattice.
    pub lattice: Lattice,
}

/// Incident plane wave. Angles are measured inside the superstrate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Excitation {
    /// Vacuum wavelength [m].
    pub wavelength: f64,
    /// Polar angle from the `z` axis [rad].
    pub theta: f64,
    /// Azimuth from the `x` axis [rad] (0 for the plane of incidence `xz`).
    pub phi: f64,
    /// Complex TE (s) amplitude.
    pub p_te: c64,
    /// Complex TM (p) amplitude.
    pub p_tm: c64,
}

impl Excitation {
    /// Pure TM (p-polarised) wave in the `xz` plane.
    pub fn tm(wavelength: f64, theta: f64) -> Self {
        Self {
            wavelength,
            theta,
            phi: 0.0,
            p_te: ZERO,
            p_tm: ONE,
        }
    }

    /// Pure TE (s-polarised) wave in the `xz` plane.
    pub fn te(wavelength: f64, theta: f64) -> Self {
        Self {
            wavelength,
            theta,
            phi: 0.0,
            p_te: ONE,
            p_tm: ZERO,
        }
    }
}

/// 2×2 block scattering matrix in the gap-medium basis.
#[derive(Clone, Debug)]
pub struct SMatrix {
    /// Reflection from above.
    pub s11: CMat,
    /// Transmission upwards.
    pub s12: CMat,
    /// Transmission downwards.
    pub s21: CMat,
    /// Reflection from below.
    pub s22: CMat,
}

impl SMatrix {
    /// Redheffer star product `self ⊗ other` (`self` above `other`).
    pub fn star(&self, other: &SMatrix) -> SMatrix {
        let n = self.s11.nrows();
        let id = eye(n);
        let d = inv(&(&id - &other.s11 * &self.s22));
        let f = inv(&(&id - &self.s22 * &other.s11));
        SMatrix {
            s11: &self.s11 + &self.s12 * &d * &other.s11 * &self.s21,
            s12: &self.s12 * &d * &other.s12,
            s21: &other.s21 * &f * &self.s21,
            s22: &other.s22 + &other.s21 * &f * &self.s22 * &other.s12,
        }
    }
}

/// Eigenmodes of one layer.
#[derive(Clone, Debug)]
struct LayerModes {
    w: CMat,
    v: CMat,
    lambda: Vec<c64>,
    e_zz_inv: CMat,
    thickness: f64,
    smatrix: SMatrix,
}

/// Diffraction efficiency of one order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrderEfficiency {
    /// Harmonic `(m, n)`.
    pub order: (i64, i64),
    /// Fraction of incident power.
    pub efficiency: f64,
    /// Whether the order propagates (else evanescent, efficiency 0).
    pub propagating: bool,
}

/// Scattering result with retained internal data for field evaluation.
#[derive(Clone, Debug)]
pub struct Scattering {
    tr: Truncation,
    k0: f64,
    kx: Vec<f64>,
    ky: Vec<f64>,
    kz_ref: Vec<c64>,
    eps_ref: c64,
    eps_trn: c64,
    v_gap: CMat,
    c_src: CMat,
    s_ref: SMatrix,
    s_trn: SMatrix,
    global: SMatrix,
    layers: Vec<LayerModes>,
    /// Reflected order efficiencies.
    pub reflection: Vec<OrderEfficiency>,
    /// Transmitted order efficiencies.
    pub transmission: Vec<OrderEfficiency>,
}

/// Complex electric field vector at a point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EField {
    /// `E_x`.
    pub x: c64,
    /// `E_y`.
    pub y: c64,
    /// `E_z`.
    pub z: c64,
}

impl EField {
    /// `|E|² / |E₀|²` (incident amplitude is unity).
    pub fn intensity(&self) -> f64 {
        self.x.norm_sqr() + self.y.norm_sqr() + self.z.norm_sqr()
    }
}

/// Which side of an interface to evaluate the discontinuous `E_z` on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// Medium above the interface (superstrate for interface 0).
    Above,
    /// Medium below the interface (layer `l`, or substrate).
    Below,
}

fn csqrt_upper(z: c64) -> c64 {
    // Principal root, then fold onto Re ≥ 0 / Im ≥ 0 for the forward wave.
    let s = z.sqrt();
    if s.im < 0.0 || (s.im == 0.0 && s.re < 0.0) {
        -s
    } else {
        s
    }
}

impl Stack {
    /// Solves the scattering problem for `exc` at truncation `tr`.
    pub fn solve(&self, exc: &Excitation, tr: Truncation) -> Scattering {
        assert!(
            self.lattice.period_y.is_some() || tr.n == 0,
            "y harmonics require a y period"
        );
        let total = tr.count();
        let k0 = 2.0 * std::f64::consts::PI / exc.wavelength;
        let n_inc = self.superstrate.sqrt();
        assert!(
            n_inc.im.abs() < 1e-12,
            "superstrate must be lossless for a well-defined incident wave"
        );
        let n_inc = n_inc.re;
        let kx0 = n_inc * exc.theta.sin() * exc.phi.cos();
        let ky0 = n_inc * exc.theta.sin() * exc.phi.sin();
        let kz_inc = n_inc * exc.theta.cos();
        let (kx, ky): (Vec<f64>, Vec<f64>) = (0..total)
            .map(|p| {
                let (m, n) = tr.orders(p);
                let gy = self
                    .lattice
                    .period_y
                    .map_or(0.0, |ly| n as f64 * exc.wavelength / ly);
                (
                    kx0 + m as f64 * exc.wavelength / self.lattice.period_x,
                    ky0 + gy,
                )
            })
            .unzip();
        let kx_m = diag_re(&kx);
        let ky_m = diag_re(&ky);
        let kxky = &kx_m * &ky_m;
        let kx2 = &kx_m * &kx_m;
        let ky2 = &ky_m * &ky_m;
        let id = eye(total);

        // Gap medium (ε = μ = 1).
        let kz_gap: Vec<c64> = (0..total)
            .map(|p| csqrt_upper(c64::new(1.0 - kx[p] * kx[p] - ky[p] * ky[p], 0.0)))
            .collect();
        let q_gap = block2(&kxky, &(&id - &kx2), &(&ky2 - &id), &(-&kxky));
        let lam_gap: Vec<c64> = kz_gap
            .iter()
            .chain(kz_gap.iter())
            .map(|k| c64::new(0.0, 1.0) * k)
            .collect();
        let v_gap = &q_gap * inv(&diag(&lam_gap));
        let w_gap = eye(2 * total);

        let half_space = |eps: c64| -> (Vec<c64>, CMat) {
            let kz: Vec<c64> = (0..total)
                .map(|p| csqrt_upper(eps - kx[p] * kx[p] - ky[p] * ky[p]))
                .collect();
            let q = block2(
                &kxky,
                &(&id * faer::Scale(eps) - &kx2),
                &(&ky2 - &id * faer::Scale(eps)),
                &(-&kxky),
            );
            let lam: Vec<c64> = kz
                .iter()
                .chain(kz.iter())
                .map(|k| c64::new(0.0, 1.0) * k)
                .collect();
            (kz, &q * inv(&diag(&lam)))
        };
        let (kz_ref, v_ref) = half_space(self.superstrate);
        let (kz_trn, v_trn) = half_space(self.substrate);

        let v_gap_inv = inv(&v_gap);
        let s_ref = {
            let a = &w_gap + &v_gap_inv * &v_ref;
            let b = &w_gap - &v_gap_inv * &v_ref;
            let a_inv = inv(&a);
            SMatrix {
                s11: -(&a_inv * &b),
                s12: &a_inv * faer::Scale(c64::new(2.0, 0.0)),
                s21: (&a - &b * &a_inv * &b) * faer::Scale(c64::new(0.5, 0.0)),
                s22: &b * &a_inv,
            }
        };
        let s_trn = {
            let a = &w_gap + &v_gap_inv * &v_trn;
            let b = &w_gap - &v_gap_inv * &v_trn;
            let a_inv = inv(&a);
            SMatrix {
                s11: &b * &a_inv,
                s12: (&a - &b * &a_inv * &b) * faer::Scale(c64::new(0.5, 0.0)),
                s21: &a_inv * faer::Scale(c64::new(2.0, 0.0)),
                s22: -(&a_inv * &b),
            }
        };

        let layers: Vec<LayerModes> = self
            .layers
            .iter()
            .map(|layer| {
                let ConvolutionMatrices { e_zz, e_xx, e_yy } =
                    layer.profile.convolution_matrices(tr);
                let e_zz_inv = inv(&e_zz);
                let p = block2(
                    &(&kx_m * &e_zz_inv * &ky_m),
                    &(&id - &kx_m * &e_zz_inv * &kx_m),
                    &(&ky_m * &e_zz_inv * &ky_m - &id),
                    &(-(&ky_m * &e_zz_inv * &kx_m)),
                );
                let q = block2(&kxky, &(&e_yy - &kx2), &(&ky2 - &e_xx), &(-&kxky));
                let omega2 = &p * &q;
                let (w, lam2) = eig(&omega2);
                // Down-going modes: λ has Re ≤ 0 (purely imaginary +jkz for
                // propagating orders in lossless media).
                let lambda: Vec<c64> = lam2
                    .iter()
                    .map(|l| {
                        let s = l.sqrt();
                        if s.re > 0.0 || (s.re == 0.0 && s.im < 0.0) {
                            -s
                        } else {
                            s
                        }
                    })
                    .collect();
                let lam_inv: Vec<c64> = lambda.iter().map(|l| ONE / l).collect();
                let v = &q * &w * diag(&lam_inv);
                let x: Vec<c64> = lambda
                    .iter()
                    .map(|l| (l * k0 * layer.thickness).exp())
                    .collect();
                let x_m = diag(&x);
                let w_inv = inv(&w);
                let v_inv = inv(&v);
                let a = &w_inv * &w_gap + &v_inv * &v_gap;
                let b = &w_inv * &w_gap - &v_inv * &v_gap;
                let a_inv = inv(&a);
                let xba = &x_m * &b * &a_inv;
                let d = &a - &xba * &x_m * &b;
                let s11 = solve(&d, &(&xba * &x_m * &a - &b));
                let s12 = solve(&d, &(&x_m * (&a - &b * &a_inv * &b)));
                LayerModes {
                    w,
                    v,
                    lambda,
                    e_zz_inv,
                    thickness: layer.thickness,
                    smatrix: SMatrix {
                        s11: s11.clone(),
                        s12: s12.clone(),
                        s21: s12,
                        s22: s11,
                    },
                }
            })
            .collect();

        let mut global = s_ref.clone();
        for l in &layers {
            global = global.star(&l.smatrix);
        }
        global = global.star(&s_trn);

        // Source polarisation vector.
        let n_hat = [0.0, 0.0, -1.0];
        let k_inc = [
            exc.theta.sin() * exc.phi.cos(),
            exc.theta.sin() * exc.phi.sin(),
            exc.theta.cos(),
        ];
        let a_te = if exc.theta.abs() < 1e-12 {
            [0.0, 1.0, 0.0]
        } else {
            normalise(cross(k_inc, n_hat))
        };
        let a_tm = normalise(cross(a_te, k_inc));
        let px = exc.p_te * a_te[0] + exc.p_tm * a_tm[0];
        let py = exc.p_te * a_te[1] + exc.p_tm * a_tm[1];
        let p0 = tr.index(0, 0);
        let c_src = Mat::from_fn(2 * total, 1, |i, _| {
            if i == p0 {
                px
            } else if i == p0 + total {
                py
            } else {
                ZERO
            }
        });
        let c_ref = &global.s11 * &c_src;
        let c_trn = &global.s21 * &c_src;
        let (rx, ry) = split_rows(&c_ref);
        let (tx, ty) = split_rows(&c_trn);
        let efficiencies = |ex: &CMat, ey: &CMat, kz: &[c64]| -> Vec<OrderEfficiency> {
            (0..total)
                .map(|p| {
                    let ez = -(kx[p] * ex[(p, 0)] + ky[p] * ey[(p, 0)]) / kz[p];
                    let mag = ex[(p, 0)].norm_sqr() + ey[(p, 0)].norm_sqr() + ez.norm_sqr();
                    let propagating = kz[p].im.abs() < 1e-12 && kz[p].re > 0.0;
                    OrderEfficiency {
                        order: tr.orders(p),
                        efficiency: if propagating {
                            kz[p].re / kz_inc * mag
                        } else {
                            0.0
                        },
                        propagating,
                    }
                })
                .collect()
        };
        let reflection = efficiencies(&rx, &ry, &kz_ref);
        let transmission = efficiencies(&tx, &ty, &kz_trn);

        Scattering {
            tr,
            k0,
            kx,
            ky,
            kz_ref,
            eps_ref: self.superstrate,
            eps_trn: self.substrate,
            v_gap,
            c_src,
            s_ref,
            s_trn,
            global,
            layers,
            reflection,
            transmission,
        }
    }
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalise(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / n, v[1] / n, v[2] / n]
}

impl Scattering {
    /// Harmonic truncation used.
    pub fn truncation(&self) -> Truncation {
        self.tr
    }

    /// Total reflectance `Σ R_mn`.
    pub fn reflectance(&self) -> f64 {
        self.reflection.iter().map(|o| o.efficiency).sum()
    }

    /// Total transmittance `Σ T_mn`.
    pub fn transmittance(&self) -> f64 {
        self.transmission.iter().map(|o| o.efficiency).sum()
    }

    /// Absorbed fraction `1 − R − T`.
    pub fn absorptance(&self) -> f64 {
        1.0 - self.reflectance() - self.transmittance()
    }

    /// Specular (zeroth-order) reflectance.
    pub fn specular_reflectance(&self) -> f64 {
        self.reflection[self.tr.index(0, 0)].efficiency
    }

    /// Zeroth-order transmittance.
    pub fn specular_transmittance(&self) -> f64 {
        self.transmission[self.tr.index(0, 0)].efficiency
    }

    /// Reflected amplitude of the specular order, `(r_x, r_y, r_z)`.
    pub fn specular_amplitude(&self) -> EField {
        let c_ref = &self.global.s11 * &self.c_src;
        let (rx, ry) = split_rows(&c_ref);
        let p = self.tr.index(0, 0);
        let x = rx[(p, 0)];
        let y = ry[(p, 0)];
        let z = -(self.kx[p] * x + self.ky[p] * y) / self.kz_ref[p];
        EField { x, y, z }
    }

    /// Global scattering matrix `S_ref ⊗ S_1 ⊗ … ⊗ S_L ⊗ S_trn`.
    pub fn global_smatrix(&self) -> &SMatrix {
        &self.global
    }

    /// Downward / upward gap-mode amplitude columns at interface `l`
    /// (interface `0` is superstrate→layer 0, interface `L` is the last
    /// layer→substrate).
    fn interface_amplitudes(&self, l: usize) -> (CMat, CMat) {
        let mut above = self.s_ref.clone();
        for layer in &self.layers[..l] {
            above = above.star(&layer.smatrix);
        }
        let mut below: Option<SMatrix> = None;
        for layer in &self.layers[l..] {
            below = Some(match below {
                None => layer.smatrix.clone(),
                Some(b) => b.star(&layer.smatrix),
            });
        }
        let below = match below {
            None => self.s_trn.clone(),
            Some(b) => b.star(&self.s_trn),
        };
        let n = above.s11.nrows();
        let d = solve(
            &(eye(n) - &above.s22 * &below.s11),
            &(&above.s21 * &self.c_src),
        );
        let u = &below.s11 * &d;
        (d, u)
    }

    /// Tangential `(s, u)` Fourier coefficient columns at interface `l`.
    fn interface_tangential(&self, l: usize) -> (CMat, CMat) {
        let (d, u) = self.interface_amplitudes(l);
        let s = &d + &u;
        let h = &self.v_gap * (&u - &d);
        (s, h)
    }

    fn real_space(&self, coeffs: &CMat, x: f64, y: f64) -> (c64, c64) {
        let total = self.tr.count();
        let mut ex = ZERO;
        let mut ey = ZERO;
        for p in 0..total {
            let phase = c64::from_polar(1.0, -self.k0 * (self.kx[p] * x + self.ky[p] * y));
            ex += coeffs[(p, 0)] * phase;
            ey += coeffs[(p + total, 0)] * phase;
        }
        (ex, ey)
    }

    /// `S_z` coefficients from tangential `H̃` and the medium's `ε_zz⁻¹`:
    /// `s_z = i ε_zz⁻¹ (Kx u_y − Ky u_x)`.
    fn sz_coeffs(&self, h: &CMat, e_zz_inv: &CMat) -> CMat {
        let total = self.tr.count();
        let rhs = Mat::from_fn(total, 1, |p, _| {
            c64::new(0.0, 1.0) * (self.kx[p] * h[(p + total, 0)] - self.ky[p] * h[(p, 0)])
        });
        e_zz_inv * rhs
    }

    /// Electric field at lateral position `(x, y)` on interface `l`, with
    /// `E_z` taken on the requested `side`.
    pub fn interface_field(&self, l: usize, side: Side, x: f64, y: f64) -> EField {
        let (s, h) = self.interface_tangential(l);
        let total = self.tr.count();
        let e_zz_inv = match side {
            Side::Above if l == 0 => &eye(total) * faer::Scale(ONE / self.eps_ref),
            Side::Above => self.layers[l - 1].e_zz_inv.clone(),
            Side::Below if l == self.layers.len() => &eye(total) * faer::Scale(ONE / self.eps_trn),
            Side::Below => self.layers[l].e_zz_inv.clone(),
        };
        let sz = self.sz_coeffs(&h, &e_zz_inv);
        let (ex, ey) = self.real_space(&s, x, y);
        let (ez, _) = self.real_space(&vstack(&sz, &sz), x, y);
        EField {
            x: ex,
            y: ey,
            z: ez,
        }
    }

    /// Electric field inside layer `l` at depth `z` below its top interface.
    pub fn layer_field(&self, l: usize, z: f64, x: f64, y: f64) -> EField {
        let layer = &self.layers[l];
        assert!(z >= 0.0 && z <= layer.thickness, "z outside layer");
        let (s, h) = self.interface_tangential(l);
        // c₋ referenced to the top: u(0) = W(c⁺ + c⁻′), h(0) = V(−c⁺ + c⁻′).
        let modes = block2(&layer.w, &layer.w, &(-&layer.v), &layer.v);
        let c = solve(&modes, &vstack(&s, &h));
        let (c_plus, c_minus_top) = split_rows(&c);
        let n2 = layer.lambda.len();
        let amp = Mat::from_fn(n2, 1, |i, _| {
            let l = layer.lambda[i];
            // Down-going modes propagate e^{λz̃} (Re λ ≤ 0 → decays into +z);
            // top-referenced up-going modes propagate e^{−λz̃} (grows into +z
            // only if reflected by the lower interface — bounded via c⁻′).
            c_plus[(i, 0)] * (l * self.k0 * z).exp()
                + c_minus_top[(i, 0)] * (-l * self.k0 * z).exp()
        });
        let amp_h = Mat::from_fn(n2, 1, |i, _| {
            let l = layer.lambda[i];
            -c_plus[(i, 0)] * (l * self.k0 * z).exp()
                + c_minus_top[(i, 0)] * (-l * self.k0 * z).exp()
        });
        let s_z = &layer.w * &amp;
        let h_z = &layer.v * &amp_h;
        let sz = self.sz_coeffs(&h_z, &layer.e_zz_inv);
        let (ex, ey) = self.real_space(&s_z, x, y);
        let (ez, _) = self.real_space(&vstack(&sz, &sz), x, y);
        EField {
            x: ex,
            y: ey,
            z: ez,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grating::Segment;
    use faer::c64;

    const LAM: f64 = 785.0e-9;

    fn half_space_stack(eps_sub: c64) -> Stack {
        Stack {
            superstrate: c64::new(1.0, 0.0),
            layers: vec![],
            substrate: eps_sub,
            lattice: Lattice {
                period_x: 960.8e-9,
                period_y: None,
            },
        }
    }

    #[test]
    fn single_interface_matches_fresnel() {
        let n2 = 1.6f64;
        for theta_deg in [0.0f64, 40.0, 67.0] {
            let theta = theta_deg.to_radians();
            let sint2 = theta.sin() / n2;
            let cost2 = (1.0 - sint2 * sint2).sqrt();
            let (c1, c2) = (theta.cos(), cost2);
            let rs = (c1 - n2 * c2) / (c1 + n2 * c2);
            let rp = (n2 * c1 - c2) / (n2 * c1 + c2);
            for (pol, expect) in [(true, rs), (false, rp)] {
                let exc = if pol {
                    Excitation::te(LAM, theta)
                } else {
                    Excitation::tm(LAM, theta)
                };
                let sc =
                    half_space_stack(c64::new(n2 * n2, 0.0)).solve(&exc, Truncation { m: 5, n: 0 });
                let a = sc.specular_amplitude();
                let got = if pol { a.y } else { -a.x / theta.cos() };
                let rel = (got - expect).norm() / expect.abs().max(1e-12);
                assert!(
                    rel < 1e-3,
                    "θ={theta_deg} pol={pol}: got {got} want {expect}"
                );
                let r = sc.specular_reflectance();
                assert!((r - expect * expect).abs() / (expect * expect).max(1e-12) < 1e-3);
            }
        }
    }

    fn airy(eps1: f64, eps2: f64, eps3: f64, d: f64, lam: f64, theta: f64, tm: bool) -> (c64, c64) {
        let (n1, n2, n3) = (eps1.sqrt(), eps2.sqrt(), eps3.sqrt());
        let c1 = theta.cos();
        let c2 = (1.0 - (n1 * theta.sin() / n2).powi(2)).sqrt();
        let c3 = (1.0 - (n1 * theta.sin() / n3).powi(2)).sqrt();
        let (adm1, adm2, adm3) = if tm {
            // Standard p-polarisation convention r_p = (n₂c₁ − n₁c₂)/(n₂c₁ + n₁c₂).
            (n1 / c1, n2 / c2, n3 / c3)
        } else {
            (n1 * c1, n2 * c2, n3 * c3)
        };
        let (r12, r23) = if tm {
            ((adm2 - adm1) / (adm1 + adm2), (adm3 - adm2) / (adm2 + adm3))
        } else {
            ((adm1 - adm2) / (adm1 + adm2), (adm2 - adm3) / (adm2 + adm3))
        };
        let beta = 2.0 * std::f64::consts::PI * n2 * d * c2 / lam;
        let ph = c64::from_polar(1.0, 2.0 * beta);
        let r = (r12 + r23 * ph) / (1.0 + r12 * r23 * ph);
        let t = (2.0 * adm2) / (adm2 + adm3) * (1.0 + r23) * ph.sqrt()
            / ((adm2 + adm1 * (r12 * ph)) / adm2).sqrt()
            * 0.0; // transmittance via 1−R
        let _ = t;
        (r, c64::new(1.0 - r.norm_sqr(), 0.0))
    }

    #[test]
    fn dielectric_slab_matches_airy_within_0_1_percent() {
        let stack = Stack {
            superstrate: c64::new(1.0, 0.0),
            layers: vec![Layer {
                thickness: 0.55 * LAM,
                profile: Profile::Uniform(c64::new(4.0, 0.0)),
            }],
            substrate: c64::new(2.25, 0.0),
            lattice: Lattice {
                period_x: 960.8e-9,
                period_y: None,
            },
        };
        for tm in [false, true] {
            for theta_deg in [0.0f64, 35.0, 60.0] {
                let theta = theta_deg.to_radians();
                let exc = if tm {
                    Excitation::tm(LAM, theta)
                } else {
                    Excitation::te(LAM, theta)
                };
                let sc = stack.solve(&exc, Truncation { m: 4, n: 0 });
                let (r_airy, _) = airy(1.0, 4.0, 2.25, 0.55 * LAM, LAM, theta, tm);
                let r_model = sc.specular_reflectance();
                let expect = r_airy.norm_sqr();
                let rel = (r_model - expect).abs() / expect.max(1e-12);
                assert!(
                    rel < 1e-3,
                    "tm={tm} θ={theta_deg}: RCWA {r_model} vs Airy {expect} ({rel:e})"
                );
                let a = sc.specular_amplitude();
                let got = if tm { -a.x / theta.cos() } else { a.y };
                assert!(
                    ((got - r_airy).norm() / r_airy.norm().max(1e-12)) < 1e-3,
                    "tm={tm} θ={theta_deg}: amplitude {got} vs {r_airy}"
                );
                assert!((sc.reflectance() + sc.transmittance() - 1.0).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn lossless_grating_conserves_energy() {
        let stack = Stack {
            superstrate: c64::new(2.113, 0.0),
            layers: vec![Layer {
                thickness: 42.5e-9,
                profile: Profile::Lamellar {
                    background: c64::new(2.113, 0.0),
                    segments: vec![Segment {
                        center: 0.5,
                        width: 0.5,
                        eps: c64::new(4.5, 0.0),
                    }],
                },
            }],
            substrate: c64::new(5.75, 0.0),
            lattice: Lattice {
                period_x: 960.8e-9,
                period_y: None,
            },
        };
        let exc = Excitation::tm(LAM, 65f64.to_radians());
        let sc = stack.solve(&exc, Truncation { m: 10, n: 0 });
        assert!((sc.reflectance() + sc.transmittance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn reflectance_converges_with_harmonics() {
        let make = |m: usize| {
            Stack {
                superstrate: c64::new(2.113, 0.0),
                layers: vec![Layer {
                    thickness: 42.5e-9,
                    profile: Profile::Lamellar {
                        background: c64::new(2.113, 0.0),
                        segments: vec![Segment {
                            center: 0.5,
                            width: 0.5,
                            eps: c64::new(-23.87, 18.99),
                        }],
                    },
                }],
                substrate: c64::new(5.75, 0.0),
                lattice: Lattice {
                    period_x: 960.8e-9,
                    period_y: None,
                },
            }
            .solve(
                &Excitation::tm(LAM, 65f64.to_radians()),
                Truncation { m, n: 0 },
            )
            .specular_reflectance()
        };
        let r = [4, 8, 12, 16].map(make);
        let tail = (r[3] - r[2]).abs();
        let head = (r[1] - r[0]).abs();
        assert!(tail < head.max(1e-12), "{r:?}");
        assert!(tail < 0.02 * r[3], "{r:?}");
    }

    #[test]
    fn tangential_fields_are_continuous_across_interface() {
        let stack = Stack {
            superstrate: c64::new(2.113, 0.0),
            layers: vec![Layer {
                thickness: 42.5e-9,
                profile: Profile::Lamellar {
                    background: c64::new(2.113, 0.0),
                    segments: vec![Segment {
                        center: 0.5,
                        width: 0.5,
                        eps: c64::new(4.5, 0.0),
                    }],
                },
            }],
            substrate: c64::new(5.75, 0.0),
            lattice: Lattice {
                period_x: 960.8e-9,
                period_y: None,
            },
        };
        let sc = stack.solve(&Excitation::tm(LAM, 0.6), Truncation { m: 8, n: 0 });
        for xf in [0.1, 0.4, 0.7] {
            let x = xf * 960.8e-9;
            let above = sc.interface_field(0, Side::Above, x, 0.0);
            let inside = sc.layer_field(0, 1e-12, x, 0.0);
            assert!((above.x - inside.x).norm() < 0.05, "{above:?} {inside:?}");
            assert!((above.y - inside.y).norm() < 0.05);
        }
    }
}
