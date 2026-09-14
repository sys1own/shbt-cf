#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    #[inline]
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline]
    pub fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    #[inline]
    pub fn one() -> Self {
        Self { re: 1.0, im: 0.0 }
    }

    #[inline]
    pub fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }

    #[inline]
    pub fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }

    #[inline]
    pub fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }

    #[inline]
    pub fn div(self, rhs: Self) -> Self {
        let denom = rhs.re * rhs.re + rhs.im * rhs.im;
        if denom == 0.0 {
            panic!("Complex division by zero");
        }

        Self::new(
            (self.re * rhs.re + self.im * rhs.im) / denom,
            (self.im * rhs.re - self.re * rhs.im) / denom,
        )
    }

    #[inline]
    pub fn norm_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    pub fn sqrt(self) -> Self {
        let magnitude = (self.re * self.re + self.im * self.im).sqrt();
        let theta = self.im.atan2(self.re);
        let root_magnitude = magnitude.sqrt();

        Self::new(
            root_magnitude * (theta * 0.5).cos(),
            root_magnitude * (theta * 0.5).sin(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::Complex64;

    #[test]
    fn complex64_basic_arithmetic() {
        let a = Complex64::new(3.0, 4.0);
        let b = Complex64::new(1.0, 2.0);

        assert_eq!(a.add(b).re, 4.0);
        assert_eq!(a.add(b).im, 6.0);
        assert_eq!(a.sub(b).re, 2.0);
        assert_eq!(a.sub(b).im, 2.0);

        let product = a.mul(b);
        assert!((product.re - (-5.0)).abs() < 1e-12);
        assert!((product.im - 10.0).abs() < 1e-12);

        let quotient = a.div(b);
        assert!((quotient.re - 2.2).abs() < 1e-12);
        assert!((quotient.im + 0.4).abs() < 1e-12);
    }

    #[test]
    fn complex64_norm_and_sqrt() {
        let a = Complex64::new(3.0, 4.0);
        assert!((a.norm_sq() - 25.0).abs() < 1e-12);

        let root = Complex64::new(1.0, 0.0).sqrt();
        assert!((root.re - 1.0).abs() < 1e-12);
        assert!((root.im).abs() < 1e-12);
    }
}
