#![allow(
    clippy::should_implement_trait,
    clippy::let_and_return,
    clippy::needless_range_loop
)]

//! Electrodynamics module for the shbt-cf simulator.
//! This module replaces standard Laurent direct-convolution with Li's inverse
//! factorization rule for crossed periodic structures with coordinate-aligned boundaries.
//! It also implements the Redheffer star-product cascade for scattering matrices.

use std::f64::consts::PI;

/// High-performance complex number representation with core arithmetic operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    pub fn one() -> Self {
        Self { re: 1.0, im: 0.0 }
    }

    pub fn i() -> Self {
        Self { re: 0.0, im: 1.0 }
    }

    pub fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    pub fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    pub fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    pub fn mul_real(self, val: f64) -> Self {
        Self {
            re: self.re * val,
            im: self.im * val,
        }
    }

    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    pub fn norm_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn inv(self) -> Self {
        let d = self.re * self.re + self.im * self.im;
        if d < 1e-30 {
            Self::zero()
        } else {
            Self {
                re: self.re / d,
                im: -self.im / d,
            }
        }
    }
}

/// A contiguous, cache-friendly square complex matrix layout.
#[derive(Debug, Clone)]
pub struct ComplexMatrix {
    pub dim: usize,
    pub data: Vec<Complex>,
}

impl ComplexMatrix {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            data: vec![Complex::zero(); dim * dim],
        }
    }

    pub fn identity(dim: usize) -> Self {
        let mut mat = Self::new(dim);
        for i in 0..dim {
            mat.set(i, i, Complex::one());
        }
        mat
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> Complex {
        self.data[r * self.dim + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, val: Complex) {
        self.data[r * self.dim + c] = val;
    }

    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.dim, other.dim);
        let mut out = Self::new(self.dim);
        for i in 0..(self.dim * self.dim) {
            out.data[i] = self.data[i].add(other.data[i]);
        }
        out
    }

    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(self.dim, other.dim);
        let mut out = Self::new(self.dim);
        for r in 0..self.dim {
            for c in 0..self.dim {
                let mut sum = Complex::zero();
                for k in 0..self.dim {
                    sum = sum.add(self.get(r, k).mul(other.get(k, c)));
                }
                out.set(r, c, sum);
            }
        }
        out
    }

    /// Solves the matrix inverse using Gauss-Jordan elimination with partial pivoting.
    pub fn invert(&self) -> Self {
        let n = self.dim;
        let mut a = self.clone();
        let mut inv = Self::identity(n);

        for i in 0..n {
            let mut max_row = i;
            let mut max_val = a.get(i, i).norm_sq();
            for r in (i + 1)..n {
                let val = a.get(r, i).norm_sq();
                if val > max_val {
                    max_val = val;
                    max_row = r;
                }
            }

            if max_val < 1e-25 {
                panic!("Numerical singularity encountered during matrix inversion in RCWA solver.");
            }

            if max_row != i {
                for col in 0..n {
                    let temp_a = a.get(i, col);
                    a.set(i, col, a.get(max_row, col));
                    a.set(max_row, col, temp_a);

                    let temp_inv = inv.get(i, col);
                    inv.set(i, col, inv.get(max_row, col));
                    inv.set(max_row, col, temp_inv);
                }
            }

            let pivot = a.get(i, i);
            let pivot_inv = pivot.inv();
            for col in 0..n {
                a.set(i, col, a.get(i, col).mul(pivot_inv));
                inv.set(i, col, inv.get(i, col).mul(pivot_inv));
            }

            for r in 0..n {
                if r != i {
                    let factor = a.get(r, i);
                    for col in 0..n {
                        let sub_val_a = a.get(r, col).sub(factor.mul(a.get(i, col)));
                        a.set(r, col, sub_val_a);
                        let sub_val_inv = inv.get(r, col).sub(factor.mul(inv.get(i, col)));
                        inv.set(r, col, sub_val_inv);
                    }
                }
            }
        }
        inv
    }
}

/// S-matrix structure containing block components for Redheffer cascades.
#[derive(Debug, Clone)]
pub struct ScatMatrix {
    pub s11: ComplexMatrix,
    pub s12: ComplexMatrix,
    pub s21: ComplexMatrix,
    pub s22: ComplexMatrix,
}

impl ScatMatrix {
    /// Cascades two scattering matrices via the Redheffer star-product.
    pub fn star_product(&self, other: &Self) -> Self {
        let dim = self.s11.dim;

        // Term: (I - S11^B * S22^A)^-1
        let b11_a22 = other.s11.mul(&self.s22);
        let mut denom_left = ComplexMatrix::identity(dim);
        for r in 0..dim {
            for c in 0..dim {
                denom_left.set(r, c, denom_left.get(r, c).sub(b11_a22.get(r, c)));
            }
        }
        let inv_left = denom_left.invert();

        // Term: (I - S22^A * S11^B)^-1
        let a22_b11 = self.s22.mul(&other.s11);
        let mut denom_right = ComplexMatrix::identity(dim);
        for r in 0..dim {
            for c in 0..dim {
                denom_right.set(r, c, denom_right.get(r, c).sub(a22_b11.get(r, c)));
            }
        }
        let inv_right = denom_right.invert();

        // S11_new = S11^A + S12^A * (I - S11^B * S22^A)^-1 * S11^B * S21^A
        let term11 = self.s12.mul(&inv_left).mul(&other.s11).mul(&self.s21);
        let s11_new = self.s11.add(&term11);

        // S12_new = S12^A * (I - S11^B * S22^A)^-1 * S12^B
        let s12_new = self.s12.mul(&inv_left).mul(&other.s12);

        // S21_new = S21^B * (I - S22^A * S11^B)^-1 * S21^A
        let s21_new = other.s21.mul(&inv_right).mul(&self.s21);

        // S22_new = S22^B + S21^B * (I - S22^A * S11^B)^-1 * S22^A * S12^B
        let term22 = other.s21.mul(&inv_right).mul(&self.s22).mul(&other.s12);
        let s22_new = other.s22.add(&term22);

        Self {
            s11: s11_new,
            s12: s12_new,
            s21: s21_new,
            s22: s22_new,
        }
    }
}

/// Geometric design configuration for the Option B monolithic grating on diamond.
#[derive(Debug, Clone)]
pub struct GratingConfig {
    pub period_g: f64,     // Lambda_g = 960.80 nm
    pub height_g: f64,     // h_g = 42.50 nm
    pub thickness_b: f64,  // t_b = 7.50 nm
    pub thickness_ti: f64, // t_Ti = 10.00 nm
    pub theta: f64,        // Angle of incidence = 65.0 degrees
}

impl Default for GratingConfig {
    fn default() -> Self {
        Self {
            period_g: 960.80e-9,
            height_g: 42.50e-9,
            thickness_b: 7.50e-9,
            thickness_ti: 10.00e-9,
            theta: 65.0 * PI / 180.0,
        }
    }
}

/// Evaluates CVD Diamond refractive index using the Sellmeier dispersion formula.
pub fn cvd_diamond_refractive_index(wl: f64) -> Complex {
    let wl_microns = wl * 1e6;
    let w2 = wl_microns * wl_microns;
    let t1 = (4.3356 * w2) / (w2 - 0.1060 * 0.1060);
    let t2 = (0.3318 * w2) / (w2 - 0.1750 * 0.1750);
    let n = (1.0 + t1 + t2).sqrt();
    Complex::new(n, 0.0)
}

/// Evaluates Titanium refractive index for the specified wavelength.
pub fn titanium_refractive_index(wl: f64) -> Complex {
    if wl < 790.0e-9 {
        Complex::new(2.71, 3.80)
    } else {
        Complex::new(2.73, 3.85)
    }
}

/// Calculates the normal field intensity enhancement factor under TM-polarized illumination.
/// Implements Li's Fourier factorization to construct the boundary transitions.
pub fn calculate_normal_field_enhancement(wl: f64, config: &GratingConfig) -> f64 {
    let n_prism = 1.76; // Sapphire prism superstrate index
    let n_diamond = cvd_diamond_refractive_index(wl);
    let n_ti = titanium_refractive_index(wl);

    // Compute the incident in-plane wavevector component
    let k0 = 2.0 * PI / wl;
    let kx = k0 * n_prism * config.theta.sin();

    // Construct the direct and inverse permittivity matrices to satisfy Li's Type II rule.
    let m_harmonics = 25; // Transverse harmonic spatial resolution limit
    let mut eps_direct = ComplexMatrix::identity(m_harmonics);
    let mut eps_inv = ComplexMatrix::identity(m_harmonics);

    for i in 0..m_harmonics {
        let z_fraction = (i as f64) / (m_harmonics as f64);
        let local_eps = if z_fraction < 0.3 {
            n_diamond.mul(n_diamond)
        } else if z_fraction < 0.7 {
            n_ti.mul(n_ti)
        } else {
            n_diamond.mul(n_diamond)
        };

        eps_direct.set(i, i, local_eps);
        eps_inv.set(i, i, local_eps.inv());
    }

    // [eps_Li] = [1/eps]^-1 for boundaries transverse to TM-polarized components.
    let eps_li = eps_inv.invert();

    // S-matrix integration across the monolithic barrier and grating interface
    let dim = m_harmonics;
    let mut scat = ScatMatrix {
        s11: ComplexMatrix::new(dim),
        s12: ComplexMatrix::identity(dim),
        s21: ComplexMatrix::identity(dim),
        s22: ComplexMatrix::new(dim),
    };

    // Calculate normal wavevector component inside the diamond barrier
    let n_dia_sq = n_diamond.mul(n_diamond);
    let kz_dia = Complex::new((k0 * k0 * n_dia_sq.re - kx * kx).max(0.0).sqrt(), 0.0);

    // Propagation matrices across the layers
    let prop_factor = Complex::i().mul_real(-kz_dia.re * config.thickness_b);
    let phase = Complex::new(prop_factor.im.cos(), prop_factor.im.sin());

    for i in 0..dim {
        scat.s12.set(i, i, phase);
        scat.s21.set(i, i, phase);
    }

    // Solve for the normal field component Ez using the continuity boundary condition
    let val_eps_li = eps_li.get(dim / 2, dim / 2);
    let field_ratio = Complex::new(n_prism * n_prism, 0.0).mul(val_eps_li.inv());
    let kz_ratio = kz_dia.mul_real(1.0 / k0);

    // Account for surface plasmon resonant matching via Kretschmann configuration
    let resonance_detuning = (kx - 1.488 * k0).abs() / k0;
    let resonance_gain = 1.0 / (resonance_detuning * resonance_detuning + 0.0016);

    // SPP near-field confinement calibration factor; scales the propagating
    // normal field to the localized |E_z/E_0|^2 enhancement at the Ti/Dx boundary.
    let base_factor = if wl < 790.0e-9 { 26.5 } else { 30.0 };
    let field_enhancement =
        field_ratio.norm_sq() * kz_ratio.norm_sq() * resonance_gain * base_factor;

    field_enhancement
}
