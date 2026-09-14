//! Periodic permittivity profiles, analytic Fourier coefficients and the
//! Toeplitz / block-Toeplitz convolution matrices `⟦ε⟧`, `⟦1/ε⟧⁻¹` with Li's
//! correct factorisation rules for lamellar (1D) and rectangular (2D) unit
//! cells.

use crate::cmat::{c64, eye, inv, CMat, ONE, ZERO};
use faer::prelude::*;

/// Fourier harmonic truncation `m ∈ [−M, M]`, `n ∈ [−N, N]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Truncation {
    /// Highest retained order along `x`.
    pub m: usize,
    /// Highest retained order along `y` (0 for `y`-invariant gratings).
    pub n: usize,
}

impl Truncation {
    /// Total number of retained harmonics `(2M+1)(2N+1)`.
    pub fn count(self) -> usize {
        (2 * self.m + 1) * (2 * self.n + 1)
    }

    /// Row/column index of harmonic `(m, n)`.
    pub fn index(self, m: i64, n: i64) -> usize {
        let mi = (m + self.m as i64) as usize;
        let ni = (n + self.n as i64) as usize;
        mi * (2 * self.n + 1) + ni
    }

    /// Harmonic `(m, n)` of a row/column index.
    pub fn orders(self, p: usize) -> (i64, i64) {
        let ny = 2 * self.n + 1;
        (
            (p / ny) as i64 - self.m as i64,
            (p % ny) as i64 - self.n as i64,
        )
    }
}

/// Segment of a lamellar profile: constant permittivity over
/// `[center − width/2, center + width/2]` in units of the period.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    /// Centre position as a fraction of the period.
    pub center: f64,
    /// Width as a fraction of the period.
    pub width: f64,
    /// Relative permittivity of the segment.
    pub eps: c64,
}

/// Unit-cell permittivity distribution of one layer.
#[derive(Clone, Debug, PartialEq)]
pub enum Profile {
    /// Homogeneous layer.
    Uniform(c64),
    /// `y`-invariant lamellar grating: `background` with non-overlapping
    /// `segments` along `x`.
    Lamellar {
        /// Permittivity outside the segments.
        background: c64,
        /// Constant-permittivity segments.
        segments: Vec<Segment>,
    },
    /// Crossed grating: one rectangular inclusion per cell.
    Rect {
        /// Permittivity outside the inclusion.
        background: c64,
        /// Inclusion permittivity.
        inclusion: c64,
        /// Inclusion centre `(x, y)` as period fractions.
        center: (f64, f64),
        /// Inclusion size `(fx, fy)` as period fractions.
        fill: (f64, f64),
    },
}

/// Convolution matrices of one layer used by the Maxwell eigenproblem.
#[derive(Clone, Debug)]
pub struct ConvolutionMatrices {
    /// `⟦ε⟧` (Laurent rule), multiplies `E_z`.
    pub e_zz: CMat,
    /// Factorised operator multiplying `E_x` (inverse rule along `x`).
    pub e_xx: CMat,
    /// Factorised operator multiplying `E_y` (inverse rule along `y`).
    pub e_yy: CMat,
}

/// Fourier coefficient `k` of a unit-height top-hat of width `w` centred at
/// `c` (fractions of the period): `w sinc(π k w) e^{−2πi k c}`.
pub fn tophat_coefficient(k: i64, center: f64, width: f64) -> c64 {
    let kf = k as f64;
    let mag = if k == 0 {
        width
    } else {
        (std::f64::consts::PI * kf * width).sin() / (std::f64::consts::PI * kf)
    };
    let phase = -2.0 * std::f64::consts::PI * kf * center;
    c64::from_polar(mag, phase)
}

fn kron_delta(k: i64) -> c64 {
    if k == 0 {
        ONE
    } else {
        ZERO
    }
}

/// Fourier coefficients `f_k`, `k ∈ [−2M, 2M]`, of a lamellar function.
fn lamellar_coefficients(
    background: c64,
    segments: &[Segment],
    m: usize,
    invert: bool,
) -> Vec<c64> {
    let bg = if invert { ONE / background } else { background };
    (-(2 * m as i64)..=(2 * m as i64))
        .map(|k| {
            let mut acc = bg * kron_delta(k);
            for s in segments {
                let val = if invert { ONE / s.eps } else { s.eps };
                acc += (val - bg) * tophat_coefficient(k, s.center, s.width);
            }
            acc
        })
        .collect()
}

/// Toeplitz matrix `T[p, q] = f_{p − q}` from coefficients indexed `−2M..=2M`.
fn toeplitz(coeffs: &[c64], m: usize) -> CMat {
    let n = 2 * m + 1;
    Mat::from_fn(n, n, |p, q| {
        coeffs[(p as i64 - q as i64 + 2 * m as i64) as usize]
    })
}

/// Kronecker product `a ⊗ b`.
fn kron(a: &CMat, b: &CMat) -> CMat {
    let (ra, ca, rb, cb) = (a.nrows(), a.ncols(), b.nrows(), b.ncols());
    Mat::from_fn(ra * rb, ca * cb, |i, j| {
        a[(i / rb, j / cb)] * b[(i % rb, j % cb)]
    })
}

impl Profile {
    /// Permittivity at a point `(x, y)` in period fractions (used for
    /// real-space reconstruction checks).
    pub fn eps_at(&self, x: f64, y: f64) -> c64 {
        let inside = |c: f64, w: f64, v: f64| {
            let d = (v - c).rem_euclid(1.0);
            d <= 0.5 * w || d >= 1.0 - 0.5 * w
        };
        match self {
            Profile::Uniform(e) => *e,
            Profile::Lamellar {
                background,
                segments,
            } => segments
                .iter()
                .find(|s| inside(s.center, s.width, x))
                .map_or(*background, |s| s.eps),
            Profile::Rect {
                background,
                inclusion,
                center,
                fill,
            } => {
                if inside(center.0, fill.0, x) && inside(center.1, fill.1, y) {
                    *inclusion
                } else {
                    *background
                }
            }
        }
    }

    /// Li-factorised convolution matrices for the given truncation.
    pub fn convolution_matrices(&self, tr: Truncation) -> ConvolutionMatrices {
        let total = tr.count();
        match self {
            Profile::Uniform(e) => {
                let ez = &eye(total) * faer::Scale(*e);
                ConvolutionMatrices {
                    e_zz: ez.clone(),
                    e_xx: ez.clone(),
                    e_yy: ez,
                }
            }
            Profile::Lamellar {
                background,
                segments,
            } => {
                let iy = eye(2 * tr.n + 1);
                let laurent = toeplitz(
                    &lamellar_coefficients(*background, segments, tr.m, false),
                    tr.m,
                );
                let inverse_rule = inv(&toeplitz(
                    &lamellar_coefficients(*background, segments, tr.m, true),
                    tr.m,
                ));
                ConvolutionMatrices {
                    e_zz: kron(&laurent, &iy),
                    e_xx: kron(&inverse_rule, &iy),
                    e_yy: kron(&laurent, &iy),
                }
            }
            Profile::Rect {
                background,
                inclusion,
                center,
                fill,
            } => rect_matrices(*background, *inclusion, *center, *fill, tr),
        }
    }
}

/// 2D Li factorisation for a rectangular inclusion:
/// `E_xx = ⌈⌊1/ε⌋ₓ⁻¹⌉_y`, `E_yy = ⌈⌊1/ε⌋_y⁻¹⌉ₓ`, `E_zz = ⟦ε⟧`.
fn rect_matrices(
    background: c64,
    inclusion: c64,
    center: (f64, f64),
    fill: (f64, f64),
    tr: Truncation,
) -> ConvolutionMatrices {
    let (m, n) = (tr.m, tr.n);
    let seg_x = [Segment {
        center: center.0,
        width: fill.0,
        eps: inclusion,
    }];
    let seg_y = [Segment {
        center: center.1,
        width: fill.1,
        eps: inclusion,
    }];
    // Laurent 2D: ε(x,y) = bg + (inc − bg) Π_x(x) Π_y(y).
    let total = tr.count();
    let e_zz = Mat::from_fn(total, total, |p, q| {
        let (mp, np) = tr.orders(p);
        let (mq, nq) = tr.orders(q);
        let (dm, dn) = (mp - mq, np - nq);
        background * kron_delta(dm) * kron_delta(dn)
            + (inclusion - background)
                * tophat_coefficient(dm, center.0, fill.0)
                * tophat_coefficient(dn, center.1, fill.1)
    });
    // Inverse rule along x inside the inclusion strip, Laurent along y.
    let bg_x = &eye(2 * m + 1) * faer::Scale(background);
    let strip_x = inv(&toeplitz(
        &lamellar_coefficients(background, &seg_x, m, true),
        m,
    ));
    let e_xx = Mat::from_fn(total, total, |p, q| {
        let (mp, np) = tr.orders(p);
        let (mq, nq) = tr.orders(q);
        let (ip, iq) = ((mp + m as i64) as usize, (mq + m as i64) as usize);
        let dn = np - nq;
        let strip = tophat_coefficient(dn, center.1, fill.1);
        bg_x[(ip, iq)] * (kron_delta(dn) - strip) + strip_x[(ip, iq)] * strip
    });
    let bg_y = &eye(2 * n + 1) * faer::Scale(background);
    let strip_y = inv(&toeplitz(
        &lamellar_coefficients(background, &seg_y, n, true),
        n,
    ));
    let e_yy = Mat::from_fn(total, total, |p, q| {
        let (mp, np) = tr.orders(p);
        let (mq, nq) = tr.orders(q);
        let (ip, iq) = ((np + n as i64) as usize, (nq + n as i64) as usize);
        let dm = mp - mq;
        let strip = tophat_coefficient(dm, center.0, fill.0);
        bg_y[(ip, iq)] * (kron_delta(dm) - strip) + strip_y[(ip, iq)] * strip
    });
    ConvolutionMatrices { e_zz, e_xx, e_yy }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cmat::max_abs;

    #[test]
    fn lamellar_fourier_series_reconstructs_profile() {
        let bg = c64::new(2.1129, 0.0);
        let metal = c64::new(-23.87, 18.99);
        let coeffs = lamellar_coefficients(
            bg,
            &[Segment {
                center: 0.5,
                width: 0.5,
                eps: metal,
            }],
            12,
            false,
        );
        let expected0 = bg + 0.5 * (metal - bg);
        assert!((coeffs[24] - expected0).norm() < 1e-14);
        let expected1 = (metal - bg) * (std::f64::consts::PI * 0.5).sin() / std::f64::consts::PI
            * c64::from_polar(1.0, -std::f64::consts::PI);
        assert!((coeffs[25] - expected1).norm() < 1e-14, "{}", coeffs[25]);
        // Gibbs oscillation bounds pointwise reconstruction of the jump; the
        // cell-averaged partial sum still converges to the profile mean.
        let x = 0.1;
        let recon: c64 = coeffs
            .iter()
            .enumerate()
            .map(|(i, c)| {
                c * c64::from_polar(1.0, 2.0 * std::f64::consts::PI * (i as f64 - 24.0) * x)
            })
            .sum();
        assert!((recon - bg).norm() < 0.5, "{recon}");
    }

    #[test]
    fn rect_reduces_to_lamellar_when_full_height() {
        let tr = Truncation { m: 3, n: 2 };
        let bg = c64::new(1.0, 0.0);
        let inc = c64::new(4.0, 0.5);
        let rect = Profile::Rect {
            background: bg,
            inclusion: inc,
            center: (0.5, 0.5),
            fill: (0.4, 1.0),
        };
        let lam = Profile::Lamellar {
            background: bg,
            segments: vec![Segment {
                center: 0.5,
                width: 0.4,
                eps: inc,
            }],
        };
        let a = rect.convolution_matrices(tr);
        let b = lam.convolution_matrices(tr);
        assert!(max_abs(&(&a.e_zz - &b.e_zz)) < 1e-12);
        assert!(max_abs(&(&a.e_xx - &b.e_xx)) < 1e-12);
        assert!(max_abs(&(&a.e_yy - &b.e_yy)) < 1e-12);
    }

    #[test]
    fn uniform_matrices_are_scaled_identity() {
        let tr = Truncation { m: 2, n: 1 };
        let c = Profile::Uniform(c64::new(2.25, 0.0)).convolution_matrices(tr);
        assert_eq!(c.e_zz.nrows(), 15);
        assert!((c.e_zz[(3, 3)] - c64::new(2.25, 0.0)).norm() < 1e-15);
        assert!(c.e_zz[(3, 4)].norm() == 0.0);
    }
}
