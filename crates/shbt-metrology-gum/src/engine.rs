//! Parallel Monte Carlo propagation of distributions per ISO/IEC Guide 98-3
//! (GUM Supplement 1): independent and covariance-correlated inputs are drawn
//! through [`Sampler`], propagated through a deterministic model, and
//! summarised as best estimate, standard uncertainty, coverage interval and
//! per-input sensitivity.

use crate::distributions::Sampler;
use crate::random::Rng;
use std::sync::Mutex;

/// Per-input sensitivity: Pearson correlation between the input draw and the
/// model output (GUM S1 §7.9 significance ordering).
#[derive(Clone, Debug, PartialEq)]
pub struct Sensitivity {
    /// Input name.
    pub name: String,
    /// `Corr(x_i, y)` in `[−1, 1]`.
    pub correlation: f64,
}

/// Result of a Monte Carlo run.
#[derive(Clone, Debug)]
pub struct McReport {
    /// Number of model evaluations.
    pub evaluations: usize,
    /// Best estimate `ȳ`.
    pub mean: f64,
    /// Standard uncertainty `u(y)`.
    pub std_dev: f64,
    /// 95 % coverage interval `(y_low, y_high)` (probabilistically symmetric).
    pub coverage_95: (f64, f64),
    /// Sensitivity per input, sorted by `|correlation|` descending.
    pub sensitivity: Vec<Sensitivity>,
    /// All model outputs (needed by callers for higher-order analysis).
    pub outputs: Vec<f64>,
}

/// Runs `n` evaluations of `model` over the sampler's distributions,
/// partitioned deterministically over `threads` workers.
///
/// Results are bit-reproducible for a given `seed`: worker `t` owns the
/// contiguous batch `[t·n/T, (t+1)·n/T)` and its private RNG sub-stream.
///
/// `model` must be thread-safe (`Sync`) and pure — it is the measurand
/// function `f(x₁ … x_d)`.
pub fn propagate<F>(sampler: &Sampler, model: F, n: usize, seed: u64, threads: usize) -> McReport
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    let threads = threads.max(1).min(n.max(1));
    let dim = sampler.dim();
    let outputs = Mutex::new(vec![0.0f64; n]);
    // Online moment/correlation accumulators, shared under a mutex.
    let acc = Mutex::new(vec![[0.0f64; 3]; dim]); // Σx, Σx², Σxy
    let model_ref = &model;
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(threads);
        for t in 0..threads {
            let acc_ref = &acc;
            let outputs_ref = &outputs;
            handles.push(scope.spawn(move || {
                let begin = t * n / threads;
                let end = (t + 1) * n / threads;
                let mut rng =
                    Rng::new(seed ^ (0x9E37_79B9_7F4A_7C15u64.wrapping_mul(t as u64 + 1)));
                let mut x = vec![0.0; dim];
                let mut local_out = Vec::with_capacity(end - begin);
                let mut local_acc = vec![[0.0f64; 3]; dim];
                for _ in begin..end {
                    sampler.draw(&mut rng, &mut x);
                    let y = model_ref(&x);
                    local_out.push(y);
                    for (i, &xi) in x.iter().enumerate() {
                        local_acc[i][0] += xi;
                        local_acc[i][1] += xi * xi;
                        local_acc[i][2] += xi * y;
                    }
                }
                outputs_ref.lock().unwrap()[begin..end].copy_from_slice(&local_out);
                let mut acc = acc_ref.lock().unwrap();
                for i in 0..dim {
                    for k in 0..3 {
                        acc[i][k] += local_acc[i][k];
                    }
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    });
    let outputs = outputs.into_inner().unwrap();
    let acc = acc.into_inner().unwrap();
    summarize(&sampler.names, outputs, acc)
}

/// Summarises raw outputs and per-input accumulators into a report.
fn summarize(names: &[String], mut outputs: Vec<f64>, acc: Vec<[f64; 3]>) -> McReport {
    let n = outputs.len();
    let nf = n as f64;
    let mean = outputs.iter().sum::<f64>() / nf;
    let var = outputs.iter().map(|y| (y - mean).powi(2)).sum::<f64>() / (nf - 1.0).max(1.0);
    let std_dev = var.sqrt();
    outputs.sort_by(f64::total_cmp);
    let lo = outputs[(0.025 * nf) as usize];
    let hi = outputs[((0.975 * nf) as usize).min(n - 1)];
    let ysum = mean * nf;
    let sensitivity: Vec<Sensitivity> = names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let sx = acc[i][0];
            let sxx = acc[i][1];
            let sxy = acc[i][2];
            let mx = sx / nf;
            let vx = (sxx - nf * mx * mx).max(0.0);
            let vy = var * (nf - 1.0).max(1.0);
            let cov = sxy - mx * ysum;
            let correlation = if vx > 0.0 && vy > 0.0 {
                cov / (vx * vy).sqrt()
            } else {
                0.0
            };
            Sensitivity {
                name: name.clone(),
                correlation,
            }
        })
        .collect();
    let mut sensitivity = sensitivity;
    sensitivity.sort_by(|a, b| b.correlation.abs().total_cmp(&a.correlation.abs()));
    McReport {
        evaluations: n,
        mean,
        std_dev,
        coverage_95: (lo, hi),
        sensitivity,
        outputs,
    }
}

/// Physical-model helpers used by the GUM uncertainty workflow
/// (cf.pdf power accounting and fatigue ceilings).
pub mod models {
    /// Net electrical export `P_net = P_TEG − P_support` [W] (cf.pdf Eq. 1):
    /// `x = [p_teg, p_support]`.
    pub fn power_net(x: &[f64]) -> f64 {
        x[0] - x[1]
    }

    /// Coffin–Manson low-cycle fatigue limit `N_f = (ε'_f/Δε_p)^{1/c}`:
    /// `x = [eps_f, c, delta_eps_p]`; returns `f64::INFINITY` for a non-positive
    /// plastic strain range.
    pub fn coffin_manson_cycles(x: &[f64]) -> f64 {
        let (eps_f, c, d_eps) = (x[0], x[1], x[2]);
        if d_eps <= 0.0 {
            f64::INFINITY
        } else {
            (eps_f / d_eps).powf(1.0 / c)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::models;
    use super::*;
    use crate::distributions::{Input, Marginal};

    #[test]
    fn normal_in_normal_out_recovers_law_of_propagation() {
        let s = Sampler::new(&[
            Input::Scalar {
                name: "a".into(),
                marginal: Marginal::Normal {
                    mean: 2.0,
                    std: 0.3,
                },
            },
            Input::Scalar {
                name: "b".into(),
                marginal: Marginal::Normal {
                    mean: 5.0,
                    std: 0.4,
                },
            },
        ])
        .unwrap();
        let r = propagate(&s, |x| 3.0 * x[0] - x[1], 200_000, 1, 4);
        // y = 3a − b ⇒ ȳ = 1, u² = 9·0.09 + 0.16 = 0.97.
        assert!((r.mean - 1.0).abs() < 0.01, "{}", r.mean);
        assert!((r.std_dev - 0.97f64.sqrt()).abs() < 0.005, "{}", r.std_dev);
        // Coverage interval ≈ ȳ ± 1.96 u.
        assert!((r.coverage_95.1 - r.coverage_95.0 - 2.0 * 1.96 * 0.97f64.sqrt()).abs() < 0.05);
        assert_eq!(r.sensitivity[0].name, "a");
        assert!(r.sensitivity[0].correlation > 0.9);
        assert!(r.sensitivity[1].correlation < -0.3);
    }

    #[test]
    fn million_evaluations_parallel_and_deterministic() {
        let s = Sampler::new(&[Input::Scalar {
            name: "x".into(),
            marginal: Marginal::Uniform { a: 0.0, b: 1.0 },
        }])
        .unwrap();
        let r1 = propagate(&s, |x| x[0] * x[0], 1_000_000, 99, 4);
        let r2 = propagate(&s, |x| x[0] * x[0], 1_000_000, 99, 4);
        assert_eq!(r1.evaluations, 1_000_000);
        assert_eq!(r1.mean.to_bits(), r2.mean.to_bits());
        // E[x²] = 1/3, Var = 4/45.
        assert!((r1.mean - 1.0 / 3.0).abs() < 0.002);
        assert!((r1.std_dev - (4.0f64 / 45.0).sqrt()).abs() < 0.002);
    }

    #[test]
    fn correlated_inputs_propagate() {
        let s = Sampler::new(&[Input::Correlated {
            names: vec!["p".into(), "s".into()],
            mean: vec![418.43, 386.67],
            covariance: vec![vec![25.0, 6.0], vec![6.0, 16.0]],
        }])
        .unwrap();
        let r = propagate(&s, models::power_net, 400_000, 7, 2);
        // P_net = p − s ⇒ Var = σp² + σs² − 2Cov = 25 + 16 − 12 = 29.
        assert!((r.mean - 31.76).abs() < 0.1, "{}", r.mean);
        assert!((r.std_dev - 29.0f64.sqrt()).abs() < 0.1, "{}", r.std_dev);
    }

    #[test]
    fn coffin_manson_limit_distribution() {
        let s = Sampler::new(&[
            Input::Scalar {
                name: "eps_f".into(),
                marginal: Marginal::Normal {
                    mean: 0.35,
                    std: 0.02,
                },
            },
            Input::Scalar {
                name: "c".into(),
                marginal: Marginal::Normal {
                    mean: 0.6,
                    std: 0.01,
                },
            },
            Input::Scalar {
                name: "dep".into(),
                marginal: Marginal::Normal {
                    mean: 4.0e-3,
                    std: 1.0e-4,
                },
            },
        ])
        .unwrap();
        let r = propagate(&s, models::coffin_manson_cycles, 200_000, 3, 4);
        // Median ≈ (0.35/0.004)^{1/0.6} ≈ 1720 cycles; heavy right tail.
        let n = r.outputs.len();
        let mut sorted = r.outputs.clone();
        sorted.sort_by(f64::total_cmp);
        let median = sorted[n / 2];
        let expect = (0.35f64 / 0.004).powf(1.0 / 0.6);
        assert!((median / expect - 1.0).abs() < 0.05, "{median} vs {expect}");
        assert!(r.coverage_95.1 > r.coverage_95.0);
    }
}
