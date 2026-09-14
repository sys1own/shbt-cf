//! PyO3 module `shbt_cf_native`: zero-copy ring access plus thin bindings to
//! the workspace solvers (simulator_spec.pdf §3, §5).
//!
//! The ring's slot memory is exposed to NumPy through
//! `PyArray1::borrow_from_array` with the `PyRing` object as the owner, so the
//! array is a direct view of the Rust allocation (and the POSIX shm segment
//! when attached) — no copy, no heap churn.

#![allow(unsafe_code)]

use crate::ring::SpscRing;
use crate::telemetry::{Channel, Frame};
use numpy::ndarray::ArrayView1;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use shbt_fea_structural::belleville::DiscSpring;
use shbt_fea_structural::material::Elastic;
use shbt_metrology_gum::distributions::{Input, Marginal, Sampler};
use shbt_metrology_gum::engine::{self, models};
use shbt_rcwa_optics::material::beat_frequency;

/// SPSC telemetry ring exposed to Python; the slot region is shared
/// zero-copy with NumPy.
#[pyclass(name = "Ring")]
pub struct PyRing {
    ring: SpscRing,
}

impl PyRing {
    fn slot_slice(&self) -> &[f64] {
        let region = self.ring.region();
        let cap = region.capacity();
        unsafe { std::slice::from_raw_parts(region.base.as_ptr().add(64) as *const f64, cap * 8) }
    }
}

#[pymethods]
impl PyRing {
    /// Heap ring with `capacity` (rounded up to a power of two) frame slots.
    #[new]
    fn new(capacity: usize) -> Self {
        Self {
            ring: SpscRing::heap(capacity.next_power_of_two()),
        }
    }

    /// Number of slots.
    fn capacity(&self) -> usize {
        self.ring.region().capacity()
    }

    /// Raw address of the frame-slot region — identical to the `.data`
    /// pointer of the array returned by [`Self::values_view`].
    fn data_ptr(&self) -> usize {
        self.slot_slice().as_ptr() as usize
    }

    /// Push `count` frames on `channel` with the given scalar value; `seq`
    /// starts at `seq0` and timestamps step at `dt_ns`.
    fn push_wave(&self, channel: u8, seq0: u64, count: usize, value: f64, dt_ns: u64) -> usize {
        let (p, _) = self.ring.split();
        let ch = decode_channel(channel);
        let mut pushed = 0usize;
        for i in 0..count {
            if !p.push(Frame::scalar(
                ch,
                seq0 + i as u64,
                (seq0 + i as u64) * dt_ns,
                value,
            )) {
                break;
            }
            pushed += 1;
        }
        pushed
    }

    /// Pop the next frame as `(channel, seq, ts_ns, values)` or `None`.
    fn pop(&self) -> Option<(u8, u64, u64, Vec<f64>)> {
        let (_, c) = self.ring.split();
        c.pop().map(|f| {
            (
                f.channel as u8,
                f.seq,
                f.timestamp_ns,
                f.values[..f.lanes as usize].to_vec(),
            )
        })
    }

    /// Frames dropped by the producer side.
    fn dropped(&self) -> usize {
        let (_, c) = self.ring.split();
        c.dropped()
    }

    /// Unread buffered frames.
    fn pending(&self) -> usize {
        let (p, _) = self.ring.split();
        p.pending()
    }

    /// **Zero-copy** `f64` view of the ring's slot memory (`capacity × 8`
    /// elements covering the packed frame array). The returned array shares
    /// the ring's allocation — `view.ctypes.data == self.data_ptr()`.
    fn values_view<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let (ptr, len) = {
            let s = slf.slot_slice();
            (s.as_ptr(), s.len())
        };
        let owner = slf.into_pyobject(py).unwrap().into_any();
        // SAFETY: the slot memory is owned by the ring inside `slf` (now the
        // array's base object) and is never reallocated while it lives.
        let view = unsafe { ArrayView1::from_shape_ptr(len, ptr) };
        unsafe { PyArray1::borrow_from_array(&view, owner) }
    }

    /// Same slot memory reshaped `(capacity, 8)` — one row per frame.
    fn frame_matrix<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let (ptr, len) = {
            let s = slf.slot_slice();
            (s.as_ptr(), s.len())
        };
        let owner = slf.into_pyobject(py).unwrap().into_any();
        // SAFETY: see `values_view`.
        let view = unsafe { ArrayView1::from_shape_ptr(len, ptr) };
        let a = unsafe { PyArray1::borrow_from_array(&view, owner) };
        let cap = a.len().unwrap() / 8;
        a.reshape([cap, 8]).unwrap()
    }
}

fn decode_channel(c: u8) -> Channel {
    match c {
        1 => Channel::ThermocoupleN,
        2 => Channel::FbgStrain,
        3 => Channel::CcdSpectrometer,
        _ => Channel::Other,
    }
}

/// `c·|λ₁⁻¹ − λ₂⁻¹|` in Hz (wavelengths in metres).
#[pyfunction]
fn beat_hz(lambda1_m: f64, lambda2_m: f64) -> f64 {
    beat_frequency(lambda1_m, lambda2_m)
}

/// DIN 2092 Belleville force (metres/pascals/newtons SI arguments).
#[pyfunction]
fn disc_force(de: f64, di: f64, t: f64, h0: f64, s: f64, e_pa: f64, nu: f64) -> f64 {
    DiscSpring {
        outer_diameter: de,
        inner_diameter: di,
        thickness: t,
        cone_height: h0,
    }
    .force(
        s,
        Elastic {
            youngs: e_pa,
            poisson: nu,
        },
    )
}

/// `χ² = rᵀ diag(σ⁻²) r`.
#[pyfunction]
fn chi2(residual: Vec<f64>, inv_var: Vec<f64>) -> f64 {
    crate::rom::chi2(&residual, &inv_var)
}

/// GUM-S1 Monte Carlo on `P_net = x₀ − x₁` with a 2×2 covariance matrix.
/// Returns `(mean, std_dev, lo95, hi95)`; `outputs` (all draws) when
/// `return_samples` is true.
#[pyfunction]
#[pyo3(signature = (mean, covariance, n, seed=0x5eed, threads=4))]
fn power_net_mc<'py>(
    py: Python<'py>,
    mean: [f64; 2],
    covariance: [[f64; 2]; 2],
    n: usize,
    seed: u64,
    threads: usize,
) -> (f64, f64, f64, f64, Bound<'py, PyArray1<f64>>) {
    let sampler = Sampler::new(&[Input::Correlated {
        names: vec!["p_teg".into(), "p_support".into()],
        mean: mean.to_vec(),
        covariance: covariance.iter().map(|r| r.to_vec()).collect(),
    }])
    .expect("covariance");
    let report =
        py.allow_threads(|| engine::propagate(&sampler, models::power_net, n, seed, threads));
    let arr = PyArray1::from_vec(py, report.outputs);
    (
        report.mean,
        report.std_dev,
        report.coverage_95.0,
        report.coverage_95.1,
        arr,
    )
}

/// Coffin–Manson low-cycle-fatigue Monte Carlo on scalar normals
/// `x = [ε′f, c, Δε_p]`; returns `(median, lo95, hi95, samples)`.
#[pyfunction]
#[pyo3(signature = (eps_f, c, delta_eps_p, n, seed=0x5eed, threads=4))]
fn coffin_manson_mc<'py>(
    py: Python<'py>,
    eps_f: (f64, f64),
    c: (f64, f64),
    delta_eps_p: (f64, f64),
    n: usize,
    seed: u64,
    threads: usize,
) -> (f64, f64, f64, Bound<'py, PyArray1<f64>>) {
    let sampler = Sampler::new(&[
        Input::Scalar {
            name: "eps_f".into(),
            marginal: Marginal::Normal {
                mean: eps_f.0,
                std: eps_f.1,
            },
        },
        Input::Scalar {
            name: "c".into(),
            marginal: Marginal::Normal {
                mean: c.0,
                std: c.1,
            },
        },
        Input::Scalar {
            name: "dep".into(),
            marginal: Marginal::Normal {
                mean: delta_eps_p.0,
                std: delta_eps_p.1,
            },
        },
    ])
    .expect("inputs");
    let report = py.allow_threads(|| {
        engine::propagate(&sampler, models::coffin_manson_cycles, n, seed, threads)
    });
    let mut sorted = report.outputs.clone();
    sorted.sort_by(f64::total_cmp);
    let arr = PyArray1::from_vec(py, report.outputs);
    (
        sorted[n / 2],
        report.coverage_95.0,
        report.coverage_95.1,
        arr,
    )
}

/// Column-wise statistics of a `n × d` matrix (used by the optimizer for
/// objective normalization).
#[pyfunction]
fn column_means(x: PyReadonlyArray2<f64>) -> Vec<f64> {
    let a = x.as_array();
    (0..a.ncols())
        .map(|j| a.column(j).mean().unwrap_or(f64::NAN))
        .collect()
}

/// `shbt_cf_native` Python module.
#[pymodule]
fn shbt_cf_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRing>()?;
    m.add_function(wrap_pyfunction!(beat_hz, m)?)?;
    m.add_function(wrap_pyfunction!(disc_force, m)?)?;
    m.add_function(wrap_pyfunction!(chi2, m)?)?;
    m.add_function(wrap_pyfunction!(power_net_mc, m)?)?;
    m.add_function(wrap_pyfunction!(coffin_manson_mc, m)?)?;
    m.add_function(wrap_pyfunction!(column_means, m)?)?;
    Ok(())
}
