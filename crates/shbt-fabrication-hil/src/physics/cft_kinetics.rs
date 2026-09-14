use rug::Float;

/// Adaptive 15-point Gauss-Kronrod reaction-rate integrator.
pub struct ConformalKineticsSolver {
    /// MPFR precision in bits.
    pub precision: u32,
    /// Absolute local quadrature tolerance.
    pub absolute_tolerance: f64,
    /// Maximum recursive subdivision depth.
    pub max_depth: usize,
}

impl ConformalKineticsSolver {
    /// Construct an integrator at the requested MPFR precision.
    pub fn new(precision: u32) -> Self {
        Self { precision, absolute_tolerance: 1e-15, max_depth: 20 }
    }

    /// Integrate `kernel_fn` over the closed energy interval.
    pub fn integrate_reaction_rate<F>(&self, kernel_fn: F, lower: &Float, upper: &Float) -> Float
    where F: Fn(&Float) -> Float {
        let (value, _) = self.integrate(&kernel_fn, lower, upper, self.max_depth);
        value
    }

    fn integrate<F>(&self, f: &F, lower: &Float, upper: &Float, depth: usize) -> (Float, Float)
    where F: Fn(&Float) -> Float {
        const X: [f64; 8] = [0.0, 0.2077849550078985, 0.4058451513773972, 0.5860872354676911,
            0.7415311855993945, 0.8648644233597691, 0.9491079123427585, 0.9914553711208126];
        const WK: [f64; 8] = [0.2094821410847278, 0.2044329400752989, 0.1903505780647854,
            0.1690047266392679, 0.1406532597155259, 0.1047900103222502,
            0.0630920926299786, 0.0229353220105292];
        const WG: [f64; 4] = [0.1294849661688697, 0.2797053914892766,
            0.3818300505051189, 0.4179591836734694];
        let half = Float::with_val(self.precision, 0.5) * Float::with_val(self.precision, upper - lower);
        let center = Float::with_val(self.precision, 0.5) * Float::with_val(self.precision, upper + lower);
        let mut kronrod = Float::with_val(self.precision, 0);
        let mut gauss = Float::with_val(self.precision, 0);
        for i in 0..8 {
            let offset = &half * Float::with_val(self.precision, X[i]);
            let x_positive = Float::with_val(self.precision, &center + &offset);
            let positive = f(&x_positive);
            let x_negative = Float::with_val(self.precision, &center - &offset);
            let negative = if i == 0 { positive.clone() } else { f(&x_negative) };
            let pair = Float::with_val(self.precision, &positive + &negative);
            if i == 0 {
                kronrod += positive.clone() * Float::with_val(self.precision, WK[i]);
            } else {
                kronrod += pair.clone() * Float::with_val(self.precision, WK[i]);
            }
            if i == 0 { gauss += positive * Float::with_val(self.precision, WG[3]); }
            else if i == 2 { gauss += Float::with_val(self.precision, &positive + &negative) * Float::with_val(self.precision, WG[2]); }
            else if i == 4 { gauss += Float::with_val(self.precision, &positive + &negative) * Float::with_val(self.precision, WG[1]); }
            else if i == 6 { gauss += Float::with_val(self.precision, &positive + &negative) * Float::with_val(self.precision, WG[0]); }
        }
        let value = kronrod.clone() * &half;
        let error = ((kronrod - gauss) * &half).abs();
        if depth == 0 || error.to_f64() <= self.absolute_tolerance { return (value, error); }
        let midpoint = center;
        let (left, left_error) = self.integrate(f, lower, &midpoint, depth - 1);
        let (right, right_error) = self.integrate(f, &midpoint, upper, depth - 1);
        (left + right, left_error + right_error)
    }
}