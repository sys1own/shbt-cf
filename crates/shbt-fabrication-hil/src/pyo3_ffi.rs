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
use numpy::ndarray::Array2;
use pyo3::prelude::*;
use shbt_fea_structural::belleville::force_per_disc;
use shbt_metrology_gum::distributions::{Input, Marginal, Sampler};
use shbt_metrology_gum::engine::{self, models};
use shbt_rcwa_optics::material::beat_frequency;
use rug::Float;

#[pyclass]
pub struct PyScreeningKernel {
    inner: crate::physics::screening::ScreeningKernel,
}

#[pymethods]
impl PyScreeningKernel {
    #[new]
    fn new(g_ef: f64, n_atom: f64, temp: f64, precision: Option<u32>) -> Self {
        let bits = precision.unwrap_or(512);
        Self { inner: crate::physics::screening::ScreeningKernel::new(
            Float::with_val(bits, g_ef), Float::with_val(bits, n_atom), Float::with_val(bits, temp),
        ) }
    }

    fn get_screening_potential(&self) -> f64 { self.inner.screening_potential.to_f64() }

    fn calculate_enhancement(&self, energy_ev: f64) -> f64 {
        let energy = Float::with_val(self.inner.screening_potential.prec(), energy_ev);
        self.inner.calculate_enhancement_factor(&energy).to_f64()
    }
}

#[pyclass(name = "ConformalKineticsSolver")]
pub struct PyConformalKineticsSolver {
    inner: crate::physics::cft_kinetics::ConformalKineticsSolver,
}

#[pymethods]
impl PyConformalKineticsSolver {
    #[new]
    fn new(precision: u32) -> Self {
        Self { inner: crate::physics::cft_kinetics::ConformalKineticsSolver::new(precision) }
    }

    /// Integrate a polynomial whose coefficients are ordered from constant to highest power.
    fn integrate_polynomial(&self, py: Python<'_>, coefficients: Vec<f64>, lower: f64, upper: f64) -> f64 {
        let solver = &self.inner;
        py.allow_threads(|| {
            let precision = solver.precision;
            let coefficients = coefficients.clone();
            let value = solver.integrate_reaction_rate(|x| {
                coefficients.iter().rev().fold(Float::with_val(precision, 0), |acc, coefficient| {
                    acc * x + Float::with_val(precision, *coefficient)
                })
            }, &Float::with_val(precision, lower), &Float::with_val(precision, upper));
            value.to_f64()
        })
    }
}

#[pyclass(name = "McNabbFosterSolver")]
pub struct PyMcNabbFosterSolver {
    inner: crate::transport::mcnabb_foster::McNabbFosterSolver,
}

#[pymethods]
impl PyMcNabbFosterSolver {
    #[new]
    #[pyo3(signature = (dx, diffusion_coeff, solubility, partial_molar_volume, c_l, hydrostatic_stress, trap_density, capture_rate, release_rate, temperature, interface_boundaries))]
    fn new(
        dx: f64,
        diffusion_coeff: Vec<f64>,
        solubility: Vec<f64>,
        partial_molar_volume: Vec<f64>,
        c_l: Vec<f64>,
        hydrostatic_stress: Vec<f64>,
        trap_density: Vec<Vec<f64>>,
        capture_rate: Vec<Vec<f64>>,
        release_rate: Vec<Vec<f64>>,
        temperature: f64,
        interface_boundaries: Vec<usize>,
    ) -> PyResult<Self> {
        let size = c_l.len();
        let valid_profiles = diffusion_coeff.len() == size
            && solubility.len() == size
            && partial_molar_volume.len() == size
            && hydrostatic_stress.len() == size;
        let valid_traps = trap_density.len() == size
            && capture_rate.len() == size
            && release_rate.len() == size
            && trap_density.iter().zip(&capture_rate).zip(&release_rate)
                .all(|((density, capture), release)| density.len() == capture.len() && density.len() == release.len());
        if dx <= 0.0 || temperature <= 0.0 || !valid_profiles || !valid_traps {
            return Err(pyo3::exceptions::PyValueError::new_err("invalid McNabb-Foster dimensions or parameters"));
        }
        let profiles = diffusion_coeff.into_iter().zip(solubility).zip(partial_molar_volume)
            .map(|((diffusion_coeff, solubility), partial_molar_volume)|
                crate::transport::mcnabb_foster::MaterialProfile { diffusion_coeff, solubility, partial_molar_volume })
            .collect();
        let trap_populations: Vec<Vec<crate::transport::mcnabb_foster::TrapState>> = trap_density.into_iter().zip(capture_rate).zip(release_rate)
            .map(|((density, capture), release)| density.into_iter().zip(capture).zip(release)
                .map(|((density, capture_rate), release_rate)| crate::transport::mcnabb_foster::TrapState { density, capture_rate, release_rate })
                .collect())
            .collect();
        let theta = trap_populations.iter().map(|traps| vec![0.0; traps.len()]).collect();
        Ok(Self { inner: crate::transport::mcnabb_foster::McNabbFosterSolver {
            grid: crate::transport::mcnabb_foster::TransportGrid { dx, size, interface_boundaries },
            profiles, trap_populations, c_l, theta, hydrostatic_stress, temperature,
        } })
    }

    fn step(&mut self, py: Python<'_>, dt: f64) -> PyResult<Vec<f64>> {
        if dt <= 0.0 { return Err(pyo3::exceptions::PyValueError::new_err("dt must be positive")); }
        py.allow_threads(|| self.inner.step(dt));
        Ok(self.inner.c_l.clone())
    }

    fn concentrations(&self) -> Vec<f64> { self.inner.c_l.clone() }
}

#[pyclass(name = "Rcwa3dSolver")]
pub struct PyRcwa3dSolver {
    inner: crate::optics::rcwa_3d::Rcwa3dSolver,
}

#[pymethods]
impl PyRcwa3dSolver {
    #[new]
    fn new(harmonics_x: usize, harmonics_y: usize, wavelength: f64) -> PyResult<Self> {
        if harmonics_x == 0 || harmonics_y == 0 || wavelength <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err("harmonics and wavelength must be positive"));
        }
        Ok(Self { inner: crate::optics::rcwa_3d::Rcwa3dSolver { harmonics_x, harmonics_y, wavelength } })
    }

    fn build_system_matrix<'py>(&self, py: Python<'py>, permittivity: PyReadonlyArray2<'py, num_complex::Complex64>) -> Bound<'py, PyArray2<num_complex::Complex64>> {
        let profile: Array2<num_complex::Complex64> = permittivity.as_array().to_owned();
        let matrix = py.allow_threads(|| self.inner.build_system_matrix(&profile));
        PyArray2::from_owned_array(py, matrix)
    }
}

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
fn disc_force(
    de: f64,
    di: f64,
    t: f64,
    h0: f64,
    deflection: f64,
    e: f64,
    nu: f64,
) -> PyResult<f64> {
    Ok(force_per_disc(de, di, t, h0, deflection, e, nu))
}

/// `χ² = rᵀ diag(σ⁻²) r`.
#[pyfunction]
fn chi2(residuals: Vec<f64>, weights: Vec<f64>) -> PyResult<f64> {
    if residuals.len() != weights.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "residuals and weights must have the same length",
        ));
    }
    Ok(crate::rom::chi2(&residuals, &weights))
}

/// GUM-S1 Monte Carlo on `P_net = x₀ − x₁` with a 2×2 covariance matrix.
/// Returns `(mean, std_dev, lo95, hi95)`; `outputs` (all draws) when
/// `return_samples` is true.
#[pyfunction]
#[pyo3(signature = (mean, cov, n, seed=0x5eed, threads=4))]
fn power_net_mc<'py>(
    py: Python<'py>,
    mean: Vec<f64>,
    cov: Vec<Vec<f64>>,
    n: usize,
    seed: u64,
    threads: usize,
) -> PyResult<(Vec<f64>, Vec<f64>, f64, f64, Vec<f64>)> {
    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n must be greater than zero",
        ));
    }
    if mean.len() != 2 || cov.len() != 2 || cov.iter().any(|row| row.len() != 2) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "mean and cov must describe two correlated inputs",
        ));
    }
    let sampler = Sampler::new(&[Input::Correlated {
        names: vec!["p_teg".into(), "p_support".into()],
        mean,
        covariance: cov,
    }])
    .map_err(|error| pyo3::exceptions::PyValueError::new_err(format!("{error:?}")))?;
    let report =
        py.allow_threads(|| engine::propagate(&sampler, models::power_net, n, seed, threads));
    Ok((
        vec![report.mean],
        vec![report.std_dev],
        report.coverage_95.0,
        report.coverage_95.1,
        report.outputs,
    ))
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
    m.add_class::<PyScreeningKernel>()?;
    m.add_class::<PyConformalKineticsSolver>()?;
    m.add_class::<PyMcNabbFosterSolver>()?;
    m.add_class::<PyRcwa3dSolver>()?;
    m.add_function(wrap_pyfunction!(beat_hz, m)?)?;
    m.add_function(wrap_pyfunction!(disc_force, m)?)?;
    m.add_function(wrap_pyfunction!(chi2, m)?)?;
    m.add_function(wrap_pyfunction!(power_net_mc, m)?)?;
    m.add_function(wrap_pyfunction!(coffin_manson_mc, m)?)?;
    m.add_function(wrap_pyfunction!(column_means, m)?)?;
    Ok(())
}
