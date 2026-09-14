//! Optical constants of the Option B stack at the two pump wavelengths
//! (cf.pdf Table VIII) and the derived complex permittivities `ε = (n + ik)²`.

use crate::cmat::c64;

/// Complex refractive index `n + ik`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RefractiveIndex {
    /// Real part.
    pub n: f64,
    /// Extinction coefficient.
    pub k: f64,
}

impl RefractiveIndex {
    /// Lossless index.
    pub const fn lossless(n: f64) -> Self {
        Self { n, k: 0.0 }
    }

    /// Relative permittivity `(n + ik)²`.
    pub fn permittivity(self) -> c64 {
        let nk = c64::new(self.n, self.k);
        nk * nk
    }
}

/// Optical constants of one Option B material at one pump wavelength.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PumpConstants {
    /// Vacuum wavelength [m].
    pub wavelength: f64,
    /// Fused-silica prism.
    pub silica: RefractiveIndex,
    /// Pd–Ir alloy relief and backplane.
    pub pd_ir: RefractiveIndex,
    /// Ti adhesion layer.
    pub ti: RefractiveIndex,
    /// CVD diamond substrate.
    pub diamond: RefractiveIndex,
}

/// Pump 1, `λ₁ = 785.0 nm` (cf.pdf Table VIII).
pub const PUMP_785: PumpConstants = PumpConstants {
    wavelength: 785.0e-9,
    silica: RefractiveIndex::lossless(1.4536),
    pd_ir: RefractiveIndex {
        n: 1.8210,
        k: 5.2140,
    },
    ti: RefractiveIndex {
        n: 2.7210,
        k: 3.7840,
    },
    diamond: RefractiveIndex::lossless(2.3980),
};

/// Pump 2, `λ₂ = 802.5 nm` (cf.pdf Table VIII).
pub const PUMP_802_5: PumpConstants = PumpConstants {
    wavelength: 802.5e-9,
    silica: RefractiveIndex::lossless(1.4531),
    pd_ir: RefractiveIndex {
        n: 1.8840,
        k: 5.3420,
    },
    ti: RefractiveIndex {
        n: 2.7650,
        k: 3.8420,
    },
    diamond: RefractiveIndex::lossless(2.3965),
};

/// Dual-pump beat frequency `c (1/λ₁ − 1/λ₂)` [Hz].
pub fn beat_frequency(lambda1: f64, lambda2: f64) -> f64 {
    299_792_458.0 * (1.0 / lambda1 - 1.0 / lambda2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beat_is_8_33_thz() {
        let f = beat_frequency(PUMP_785.wavelength, PUMP_802_5.wavelength);
        assert!((f - 8.328_064e12).abs() < 1e7, "{f}");
    }

    #[test]
    fn metal_permittivity_has_negative_real_part() {
        let e = PUMP_785.pd_ir.permittivity();
        assert!(e.re < 0.0 && e.im > 0.0);
        assert!((e.re - (1.8210f64.powi(2) - 5.2140f64.powi(2))).abs() < 1e-12);
    }
}
