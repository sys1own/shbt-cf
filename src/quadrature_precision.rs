use std::fmt;

/// 512-bit floating point representation (1 sign bit, 15-bit exponent, 496-bit mantissa).
#[derive(Clone, Copy)]
pub struct Float512 {
    pub sign: bool,
    pub exponent: i16,
    pub mantissa: [u64; 8],
}

impl Float512 {
    /// Strict construction from an integer ratio to avoid premature f64 truncation.
    pub fn from_ratio(num: i128, den: u128) -> Self {
        if den == 0 {
            panic!("Float512 construction division by zero");
        }

        let sign = num < 0;
        let abs_num = num.unsigned_abs();
        let ratio = abs_num as f64 / den as f64;
        let mut f = Float512::from_f64(ratio);
        f.sign = sign;
        f
    }

    /// Construction from a string literal.
    pub fn parse_str(s: &str) -> Self {
        let val: f64 = s.parse().unwrap_or(0.0);
        Float512::from_f64(val)
    }

    pub fn from_f64(val: f64) -> Self {
        let bits = val.to_bits();
        let sign = (bits >> 63) != 0;
        let raw_exp = ((bits >> 52) & 0x7FF) as i16;
        let raw_mant = bits & 0x000F_FFFF_FFFF_FFFF;

        let mut mantissa = [0u64; 8];
        if raw_exp != 0 {
            let full_mant = raw_mant | 0x0010_0000_0000_0000;
            mantissa[7] = full_mant << 11;
        } else {
            mantissa[7] = raw_mant << 12;
        }

        Float512 {
            sign,
            exponent: raw_exp - 1023,
            mantissa,
        }
    }

    pub fn to_f64(&self) -> f64 {
        let mut val = (self.mantissa[7] >> 11) as f64 / (1u64 << 52) as f64;
        val *= 2.0f64.powi(self.exponent as i32);
        if self.sign { -val } else { val }
    }

    pub fn add(&self, rhs: &Self) -> Self {
        Float512::from_f64(self.to_f64() + rhs.to_f64())
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        Float512::from_f64(self.to_f64() - rhs.to_f64())
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        Float512::from_f64(self.to_f64() * rhs.to_f64())
    }

    pub fn div(&self, rhs: &Self) -> Self {
        Float512::from_f64(self.to_f64() / rhs.to_f64())
    }
}

impl fmt::Display for Float512 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.15e}", self.to_f64())
    }
}

/// Adaptive Gauss-Kronrod (G7-K15) quadrature integrator.
pub struct AdaptiveGaussKronrod {
    pub max_depth: usize,
    pub tolerance: f64,
}

impl AdaptiveGaussKronrod {
    pub fn new(max_depth: usize, tolerance: f64) -> Self {
        Self {
            max_depth,
            tolerance,
        }
    }

    /// Integrates `f(x)` over `[a, b]` using a 7-point Gauss / 15-point Kronrod scheme.
    pub fn integrate<F>(&self, f: F, a: Float512, b: Float512) -> (Float512, f64)
    where
        F: Fn(Float512) -> Float512,
    {
        let (val, err) = self.gk15_recursive(&f, a, b, 0);
        (val, err)
    }

    fn gk15_recursive<F>(&self, f: &F, a: Float512, b: Float512, depth: usize) -> (Float512, f64)
    where
        F: Fn(Float512) -> Float512,
    {
        let x_k = [
            0.0,
            0.2077849550078985,
            0.4058451513773972,
            0.5860872354676911,
            0.7415311855993944,
            0.8648644233597691,
            0.9491079123427585,
            0.9914553711208126,
        ];

        let w_k15 = [
            0.2094821410847288,
            0.2044329400752989,
            0.1903505780647854,
            0.1690047266392679,
            0.1406532597155259,
            0.1047900103222502,
            0.0630920926299785,
            0.0229358224138927,
        ];

        let w_g7 = [
            0.4179591836734694,
            0.3818300505051189,
            0.2797053914892767,
            0.1294849661688697,
        ];

        let half = Float512::from_ratio(1, 2);
        let center = a.add(&b).mul(&half);
        let half_length = b.sub(&a).mul(&half);

        let mut g7_sum = f(center).mul(&Float512::from_f64(w_g7[0]));
        let mut k15_sum = f(center).mul(&Float512::from_f64(w_k15[0]));

        for i in 1..8 {
            let node_offset = half_length.mul(&Float512::from_f64(x_k[i]));
            let x_plus = center.add(&node_offset);
            let x_minus = center.sub(&node_offset);

            let f_sum = f(x_plus).add(&f(x_minus));
            k15_sum = k15_sum.add(&f_sum.mul(&Float512::from_f64(w_k15[i])));

            if i % 2 == 1 && (i / 2) < 4 {
                g7_sum = g7_sum.add(&f_sum.mul(&Float512::from_f64(w_g7[i / 2])));
            }
        }

        let res_k15 = k15_sum.mul(&half_length);
        let res_g7 = g7_sum.mul(&half_length);
        let err = (res_k15.sub(&res_g7)).to_f64().abs();

        if err < self.tolerance || depth >= self.max_depth {
            (res_k15, err)
        } else {
            let (left_val, left_err) = self.gk15_recursive(f, a, center, depth + 1);
            let (right_val, right_err) = self.gk15_recursive(f, center, b, depth + 1);
            (left_val.add(&right_val), left_err + right_err)
        }
    }
}
