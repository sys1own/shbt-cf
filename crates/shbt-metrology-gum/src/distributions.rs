//! Input probability distributions for GUM Supplement 1 sampling:
//! independent marginals plus multivariate-Gaussian blocks with a full
//! covariance matrix ingested via its Cholesky factor.

use crate::random::Rng;

/// Marginal distribution of one scalar input quantity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Marginal {
    /// Gaussian `N(μ, σ²)`.
    Normal {
        /// Best estimate.
        mean: f64,
        /// Standard uncertainty.
        std: f64,
    },
    /// Rectangular `U(a, b)` (symmetric tolerance without a distribution model).
    Uniform {
        /// Lower bound.
        a: f64,
        /// Upper bound.
        b: f64,
    },
    /// Log-normal: `ln X ~ N(μ_ln, σ_ln²)`.
    LogNormal {
        /// Log-space mean.
        mean_ln: f64,
        /// Log-space standard deviation.
        std_ln: f64,
    },
}

/// A block of input quantities: either a scalar marginal or a multivariate
/// Gaussian block described by its full covariance matrix.
#[derive(Clone, Debug)]
pub enum Input {
    /// One independent scalar input.
    Scalar {
        /// Quantity name (carried into sensitivity reports).
        name: String,
        /// Marginal distribution.
        marginal: Marginal,
    },
    /// A `d`-dimensional correlated Gaussian block `N(μ, Σ)`.
    Correlated {
        /// Quantity names (length `d`).
        names: Vec<String>,
        /// Best estimates `μ`.
        mean: Vec<f64>,
        /// Full covariance matrix `Σ` (`d × d`, symmetric positive
        /// semi-definite), ingested from calibration data.
        covariance: Vec<Vec<f64>>,
    },
}

/// Errors building an input model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputError {
    /// Covariance matrix not square / inconsistent dimension.
    Dimension,
    /// Covariance matrix not positive definite (zero allowed on request).
    NotPositiveDefinite,
}

/// Dense `n × n` lower-triangular Cholesky factor `L` with `LLᵀ = Σ`.
pub fn cholesky(sigma: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, InputError> {
    let n = sigma.len();
    if sigma.iter().any(|r| r.len() != n) {
        return Err(InputError::Dimension);
    }
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = sigma[i][j];
            for (lk, rk) in l[i].iter().zip(l[j].iter()).take(j) {
                s -= lk * rk;
            }
            l[i][j] = if i == j {
                if s < -1e-14 * sigma[i][i].abs().max(1.0) {
                    return Err(InputError::NotPositiveDefinite);
                }
                s.max(0.0).sqrt()
            } else {
                s / l[j][j].max(f64::MIN_POSITIVE)
            };
        }
    }
    Ok(l)
}

/// Prepared sampler producing one input vector per draw.
#[derive(Clone, Debug)]
pub struct Sampler {
    blocks: Vec<Block>,
    dim: usize,
    /// Input names in model-vector order.
    pub names: Vec<String>,
}

#[derive(Clone, Debug)]
enum Block {
    Scalar(usize, Marginal),
    Correlated {
        first: usize,
        lower: Vec<Vec<f64>>,
        mean: Vec<f64>,
    },
}

impl Sampler {
    /// Builds a sampler, validating and factorising covariance blocks.
    pub fn new(inputs: &[Input]) -> Result<Self, InputError> {
        let mut blocks = Vec::with_capacity(inputs.len());
        let mut names = Vec::new();
        for input in inputs {
            match input {
                Input::Scalar { name, marginal } => {
                    blocks.push(Block::Scalar(names.len(), *marginal));
                    names.push(name.clone());
                }
                Input::Correlated {
                    names: ns,
                    mean,
                    covariance,
                } => {
                    if ns.len() != mean.len() {
                        return Err(InputError::Dimension);
                    }
                    let lower = cholesky(covariance)?;
                    blocks.push(Block::Correlated {
                        first: names.len(),
                        lower,
                        mean: mean.clone(),
                    });
                    names.extend(ns.iter().cloned());
                }
            }
        }
        let dim = names.len();
        Ok(Self { blocks, dim, names })
    }

    /// Model input vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Draws one input vector into `out`.
    pub fn draw(&self, rng: &mut Rng, out: &mut [f64]) {
        for block in &self.blocks {
            match block {
                Block::Scalar(i, m) => {
                    out[*i] = match m {
                        Marginal::Normal { mean, std } => mean + std * rng.normal(),
                        Marginal::Uniform { a, b } => a + (b - a) * rng.uniform(),
                        Marginal::LogNormal { mean_ln, std_ln } => {
                            (mean_ln + std_ln * rng.normal()).exp()
                        }
                    };
                }
                Block::Correlated { first, lower, mean } => {
                    let d = mean.len();
                    let z: Vec<f64> = (0..d).map(|_| rng.normal()).collect();
                    for i in 0..d {
                        let mut v = mean[i];
                        for (k, &zk) in z.iter().enumerate().take(i + 1) {
                            v += lower[i][k] * zk;
                        }
                        out[first + i] = v;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_factorises_covariance() {
        let s = vec![vec![4.0, 1.2], vec![1.2, 1.0]];
        let l = cholesky(&s).unwrap();
        let prod = |i: usize, j: usize| (0..=i.min(j)).map(|k| l[i][k] * l[j][k]).sum::<f64>();
        assert!((prod(0, 0) - 4.0).abs() < 1e-14);
        assert!((prod(0, 1) - 1.2).abs() < 1e-14);
        assert!((prod(1, 1) - 1.0).abs() < 1e-14);
        assert_eq!(
            cholesky(&[vec![1.0, 2.0], vec![2.0, 1.0]]),
            Err(InputError::NotPositiveDefinite)
        );
    }

    #[test]
    fn correlated_block_recovers_covariance() {
        let s = Sampler::new(&[
            Input::Correlated {
                names: vec!["x".into(), "y".into()],
                mean: vec![10.0, -5.0],
                covariance: vec![vec![9.0, 2.4], vec![2.4, 4.0]],
            },
            Input::Scalar {
                name: "u".into(),
                marginal: Marginal::Uniform { a: -1.0, b: 1.0 },
            },
        ])
        .unwrap();
        let mut rng = Rng::new(1234);
        let n = 400_000usize;
        let (mut mx, mut my, mut cxy, mut vy) = (0.0, 0.0, 0.0, 0.0);
        let mut out = vec![0.0; s.dim()];
        for _ in 0..n {
            s.draw(&mut rng, &mut out);
            mx += out[0] / n as f64;
            my += out[1] / n as f64;
        }
        rng = Rng::new(1234);
        for _ in 0..n {
            s.draw(&mut rng, &mut out);
            cxy += (out[0] - mx) * (out[1] - my) / n as f64;
            vy += (out[1] - my).powi(2) / n as f64;
        }
        assert!((mx - 10.0).abs() < 0.02, "{mx}");
        assert!((my + 5.0).abs() < 0.02, "{my}");
        assert!((cxy - 2.4).abs() < 0.05, "{cxy}");
        assert!((vy - 4.0).abs() < 0.05, "{vy}");
    }
}
