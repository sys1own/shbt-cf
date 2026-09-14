//! Forward-mode automatic differentiation and first-order GUM propagation.

/// A scalar value with its gradient with respect to the independent inputs.
#[derive(Clone, Debug, PartialEq)]
pub struct DualNum {
    /// Nominal value.
    pub value: f64,
    /// Partial derivatives in input order.
    pub derivatives: Vec<f64>,
}

impl DualNum {
    /// Creates an independent input variable.
    pub fn variable(value: f64, index: usize, dimension: usize) -> Self {
        let mut derivatives = vec![0.0; dimension];
        derivatives[index] = 1.0;
        Self { value, derivatives }
    }

    /// Creates a constant in a gradient space.
    pub fn constant(value: f64, dimension: usize) -> Self {
        Self {
            value,
            derivatives: vec![0.0; dimension],
        }
    }

    /// Exponential function.
    pub fn exp(self) -> Self {
        let scale = self.value.exp();
        Self {
            value: scale,
            derivatives: self.derivatives.into_iter().map(|d| scale * d).collect(),
        }
    }

    /// Natural logarithm.
    pub fn ln(self) -> Self {
        let value = self.value.ln();
        Self {
            value,
            derivatives: self
                .derivatives
                .into_iter()
                .map(|d| d / self.value)
                .collect(),
        }
    }

    /// Power with a constant exponent.
    pub fn powf(self, exponent: f64) -> Self {
        let value = self.value.powf(exponent);
        let scale = exponent * self.value.powf(exponent - 1.0);
        Self {
            value,
            derivatives: self.derivatives.into_iter().map(|d| scale * d).collect(),
        }
    }
}

impl std::ops::Add for DualNum {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        debug_assert_eq!(self.derivatives.len(), rhs.derivatives.len());
        Self {
            value: self.value + rhs.value,
            derivatives: self
                .derivatives
                .into_iter()
                .zip(rhs.derivatives)
                .map(|(a, b)| a + b)
                .collect(),
        }
    }
}

impl std::ops::Sub for DualNum {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        debug_assert_eq!(self.derivatives.len(), rhs.derivatives.len());
        Self {
            value: self.value - rhs.value,
            derivatives: self
                .derivatives
                .into_iter()
                .zip(rhs.derivatives)
                .map(|(a, b)| a - b)
                .collect(),
        }
    }
}

impl std::ops::Mul for DualNum {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        debug_assert_eq!(self.derivatives.len(), rhs.derivatives.len());
        let left = self.value;
        let right = rhs.value;
        Self {
            value: left * right,
            derivatives: self
                .derivatives
                .into_iter()
                .zip(rhs.derivatives)
                .map(|(a, b)| a * right + b * left)
                .collect(),
        }
    }
}

impl std::ops::Div for DualNum {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        debug_assert_eq!(self.derivatives.len(), rhs.derivatives.len());
        let denominator = rhs.value * rhs.value;
        let left = self.value;
        let right = rhs.value;
        Self {
            value: left / right,
            derivatives: self
                .derivatives
                .into_iter()
                .zip(rhs.derivatives)
                .map(|(a, b)| (a * right - b * left) / denominator)
                .collect(),
        }
    }
}

/// Result of GUM covariance propagation.
#[derive(Clone, Debug, PartialEq)]
pub struct GumReport {
    /// Output nominal values.
    pub values: Vec<f64>,
    /// Jacobian, indexed by output then input.
    pub jacobian: Vec<Vec<f64>>,
    /// Output covariance matrix `J Σ Jᵀ`.
    pub covariance: Vec<Vec<f64>>,
}

/// Propagates an input covariance matrix through a dual-number model.
pub fn propagate_uncertainty_gum<F>(means: &[f64], covariance: &[Vec<f64>], model: F) -> GumReport
where
    F: Fn(&[DualNum]) -> Vec<DualNum>,
{
    let n = means.len();
    assert_eq!(covariance.len(), n, "covariance dimension mismatch");
    assert!(
        covariance.iter().all(|row| row.len() == n),
        "covariance must be square"
    );
    let inputs: Vec<_> = means
        .iter()
        .enumerate()
        .map(|(i, &x)| DualNum::variable(x, i, n))
        .collect();
    let outputs = model(&inputs);
    let values = outputs.iter().map(|output| output.value).collect();
    let jacobian = outputs
        .iter()
        .map(|output| output.derivatives.clone())
        .collect::<Vec<_>>();
    let mut result = vec![vec![0.0; outputs.len()]; outputs.len()];
    for (i, left) in jacobian.iter().enumerate() {
        for (j, right) in jacobian.iter().enumerate() {
            result[i][j] = left
                .iter()
                .enumerate()
                .map(|(a, &da)| {
                    da * right
                        .iter()
                        .enumerate()
                        .map(|(b, &db)| covariance[a][b] * db)
                        .sum::<f64>()
                })
                .sum();
        }
    }
    GumReport {
        values,
        jacobian,
        covariance: result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derivatives_and_covariance_are_analytic() {
        let report =
            propagate_uncertainty_gum(&[2.0, 3.0], &[vec![0.25, 0.1], vec![0.1, 0.36]], |x| {
                vec![x[0].clone() * x[1].clone(), x[0].clone() + x[1].clone()]
            });
        assert_eq!(report.values, vec![6.0, 5.0]);
        assert_eq!(report.jacobian, vec![vec![3.0, 2.0], vec![1.0, 1.0]]);
        assert!((report.covariance[0][0] - 4.89).abs() < 1e-12);
        assert!((report.covariance[1][1] - 0.81).abs() < 1e-12);
    }
}
