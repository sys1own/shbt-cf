//! Python bindings for the transport and coherence kernels.
#![allow(missing_docs)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use shbt_rcwa::RcwaSystemSolver;

const R_GAS: f64 = 8.31446261815324;
const PLANCK_REDUCED: f64 = 1.054571817e-34;
const BOLTZMANN: f64 = 1.380649e-23;

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyTransportConfig {
    #[pyo3(get, set)]
    pub partial_molar_volume: f64,
    #[pyo3(get, set)]
    pub soret_heat: f64,
    #[pyo3(get, set)]
    pub lattice_site_density: f64,
    #[pyo3(get, set)]
    pub diffusion_coefficient: f64,
    #[pyo3(get, set)]
    pub activation_energy: f64,
    #[pyo3(get, set)]
    pub reference_temperature: f64,
}

#[pymethods]
impl PyTransportConfig {
    #[new]
    #[pyo3(signature = (partial_molar_volume=0.0, soret_heat=0.0, lattice_site_density=1.0, diffusion_coefficient=1.0e-9, activation_energy=0.0, reference_temperature=293.0))]
    fn new(
        partial_molar_volume: f64,
        soret_heat: f64,
        lattice_site_density: f64,
        diffusion_coefficient: f64,
        activation_energy: f64,
        reference_temperature: f64,
    ) -> PyResult<Self> {
        validate_transport_config(
            partial_molar_volume,
            lattice_site_density,
            diffusion_coefficient,
            reference_temperature,
        )?;
        Ok(Self {
            partial_molar_volume,
            soret_heat,
            lattice_site_density,
            diffusion_coefficient,
            activation_energy,
            reference_temperature,
        })
    }

    fn diffusion_at_temperature(&self, temperature: f64) -> PyResult<f64> {
        if temperature <= 0.0 {
            return Err(PyValueError::new_err("temperature must be positive"));
        }
        Ok(self.diffusion_coefficient * (-self.activation_energy / (R_GAS * temperature)).exp())
    }
}

#[pyclass]
pub struct PyQuantumCoherenceSolver {
    #[pyo3(get, set)]
    pub larmor_frequency: f64,
    #[pyo3(get, set)]
    pub coupling_constant: f64,
    #[pyo3(get, set)]
    pub base_dephasing_rate: f64,
    #[pyo3(get, set)]
    pub reference_temperature: f64,
}

#[pymethods]
impl PyQuantumCoherenceSolver {
    #[new]
    #[pyo3(signature = (larmor_frequency, coupling_constant, base_dephasing_rate=1.0e-4, reference_temperature=293.0))]
    fn new(
        larmor_frequency: f64,
        coupling_constant: f64,
        base_dephasing_rate: f64,
        reference_temperature: f64,
    ) -> PyResult<Self> {
        if larmor_frequency < 0.0 || base_dephasing_rate < 0.0 || reference_temperature <= 0.0 {
            return Err(PyValueError::new_err("invalid coherence parameters"));
        }
        Ok(Self {
            larmor_frequency,
            coupling_constant,
            base_dephasing_rate,
            reference_temperature,
        })
    }

    fn dephasing_rate(&self, temperature: f64) -> PyResult<f64> {
        if temperature <= 0.0 {
            return Err(PyValueError::new_err("temperature must be positive"));
        }
        Ok(self.base_dephasing_rate * (temperature / self.reference_temperature).powi(3))
    }

    fn solve_coherence_decay(&self, temperature: f64, time_steps: Vec<f64>) -> PyResult<Vec<f64>> {
        let rate = self.dephasing_rate(temperature)?;
        if time_steps.iter().any(|time| *time < 0.0) {
            return Err(PyValueError::new_err("time steps must be non-negative"));
        }
        Ok(time_steps
            .into_iter()
            .map(|time| (-rate * time).exp())
            .collect())
    }

    fn thermal_variance(&self, temperature: f64) -> PyResult<f64> {
        if temperature <= 0.0 {
            return Err(PyValueError::new_err("temperature must be positive"));
        }
        Ok(BOLTZMANN * temperature / (self.coupling_constant.abs().max(1e-30)))
    }

    fn franck_condon_rate(
        &self,
        energy: f64,
        temperature: f64,
        phonon_frequency: f64,
        huang_rhys: f64,
        periods: usize,
        samples_per_period: usize,
    ) -> PyResult<f64> {
        if temperature <= 0.0
            || phonon_frequency <= 0.0
            || huang_rhys < 0.0
            || periods == 0
            || samples_per_period < 4
        {
            return Err(PyValueError::new_err("invalid Franck-Condon parameters"));
        }
        let period = 2.0 * std::f64::consts::PI / phonon_frequency;
        let count = periods * samples_per_period;
        let step = periods as f64 * period / count as f64;
        let occupancy = 1.0
            / ((PLANCK_REDUCED * phonon_frequency / (BOLTZMANN * temperature)).exp() - 1.0)
                .max(1e-15);
        let mut integral = 0.0;
        for index in 0..=count {
            let time = index as f64 * step;
            let envelope = (-huang_rhys
                * ((2.0 * occupancy + 1.0)
                    - 2.0 * occupancy * (phonon_frequency * time).cos()
                    - (phonon_frequency * time).cos()))
            .exp();
            let value = envelope * (energy * time / PLANCK_REDUCED).cos();
            let weight = if index == 0 || index == count {
                1.0
            } else if index % 2 == 0 {
                2.0
            } else {
                4.0
            };
            integral += weight * value;
        }
        Ok((integral * step / 3.0 / PLANCK_REDUCED.powi(2)).max(0.0))
    }
}

#[pyfunction]
#[pyo3(signature = (config, lattice_init, steps, dt=1.0e-3, temperature=293.0, interface_boundaries=None, segregation_factors=None))]
fn solve_mcnabb_foster_transport(
    config: PyTransportConfig,
    lattice_init: Vec<f64>,
    steps: usize,
    dt: f64,
    temperature: f64,
    interface_boundaries: Option<Vec<usize>>,
    segregation_factors: Option<Vec<f64>>,
) -> PyResult<Vec<f64>> {
    if lattice_init.len() < 2
        || dt <= 0.0
        || temperature <= 0.0
        || lattice_init
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(PyValueError::new_err("invalid transport state or timestep"));
    }
    let boundaries = interface_boundaries.unwrap_or_default();
    if boundaries
        .iter()
        .any(|index| *index == 0 || *index >= lattice_init.len())
    {
        return Err(PyValueError::new_err(
            "interface boundaries must be interior nodes",
        ));
    }
    let factors = segregation_factors.unwrap_or_else(|| vec![1.0; boundaries.len()]);
    if factors.len() != boundaries.len()
        || factors
            .iter()
            .any(|factor| !factor.is_finite() || *factor <= 0.0)
    {
        return Err(PyValueError::new_err(
            "segregation factors must match interfaces",
        ));
    }
    let diffusion = config.diffusion_at_temperature(temperature)?;
    let mut state = lattice_init;
    let dx = 1.0 / (state.len() - 1) as f64;
    let courant = diffusion * dt / dx.powi(2);
    if courant > 0.5 {
        return Err(PyValueError::new_err(
            "explicit transport stability limit exceeded",
        ));
    }
    for _ in 0..steps {
        let old = state.clone();
        for index in 1..state.len() - 1 {
            let laplacian = (old[index + 1] - 2.0 * old[index] + old[index - 1]) / dx.powi(2);
            let soret =
                config.soret_heat * old[index] * (temperature - config.reference_temperature)
                    / (R_GAS * temperature.powi(2));
            state[index] = old[index] + dt * diffusion * (laplacian - soret);
        }
        for (position, boundary) in boundaries.iter().enumerate() {
            let left = state[*boundary - 1];
            let right = state[*boundary];
            let target = factors[position] * left;
            state[*boundary] = 0.5 * right + 0.5 * target;
        }
        for value in &mut state {
            *value = value.max(0.0);
        }
    }
    Ok(state)
}

fn validate_transport_config(
    partial_molar_volume: f64,
    lattice_site_density: f64,
    diffusion_coefficient: f64,
    reference_temperature: f64,
) -> PyResult<()> {
    if !partial_molar_volume.is_finite()
        || lattice_site_density <= 0.0
        || diffusion_coefficient < 0.0
        || reference_temperature <= 0.0
    {
        return Err(PyValueError::new_err("invalid transport configuration"));
    }
    Ok(())
}

#[pyfunction]
fn solve_2d_rcwa_identity(
    harmonics_m: usize,
    harmonics_n: usize,
    wavelength: f64,
) -> PyResult<(usize, f64, f64)> {
    if harmonics_m == 0 || harmonics_n == 0 || wavelength <= 0.0 {
        return Err(PyValueError::new_err(
            "harmonics and wavelength must be positive",
        ));
    }
    let solver = RcwaSystemSolver::with_harmonics(harmonics_m, harmonics_n, wavelength);
    let scattering = solver.identity_s_matrix();
    Ok((
        solver.mode_count(),
        scattering.s11[[0, 0]].re,
        scattering.s22[[0, 0]].re,
    ))
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn belleville_thermal_compliance(
    outer_radius: f64,
    inner_radius: f64,
    thickness: f64,
    youngs_modulus: f64,
    poissons_ratio: f64,
    temperature: f64,
    thermal_expansion_coeff: f64,
    reference_temperature: f64,
    force: f64,
) -> PyResult<(f64, f64)> {
    if outer_radius <= inner_radius
        || inner_radius <= 0.0
        || thickness <= 0.0
        || youngs_modulus <= 0.0
        || !(0.0..0.5).contains(&poissons_ratio)
        || temperature <= 0.0
    {
        return Err(PyValueError::new_err(
            "invalid Belleville geometry or material parameters",
        ));
    }
    let span = outer_radius - inner_radius;
    let modulus = youngs_modulus * (1.0 - 0.0005 * (temperature - reference_temperature)).max(0.01);
    let stiffness =
        modulus * thickness.powi(3) / (12.0 * (1.0 - poissons_ratio.powi(2)) * span.powi(2));
    let thermal_deflection = thermal_expansion_coeff * (temperature - reference_temperature) * span;
    Ok((force / stiffness + thermal_deflection, stiffness))
}

/// Pd_{0.9132}Ir_{0.0868} Chaboche model mirror (update-11.1 Task 3).
///
/// Mirrors `shbt_fea_structural::viscoplastic::ChabocheViscoplasticModel`;
/// duplicated here because the workspace topology fixes this crate's
/// dependency set to `shbt-rcwa` only (see `scripts/check_dep_graph.py`).
#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct PyChabocheSolver {
    #[pyo3(get)]
    pub k_visco: f64,
    #[pyo3(get)]
    pub n_visco: f64,
    #[pyo3(get)]
    pub q_iso: f64,
    #[pyo3(get)]
    pub b_iso: f64,
    #[pyo3(get)]
    pub gamma1: f64,
    #[pyo3(get)]
    pub gamma2: f64,
}

#[pymethods]
impl PyChabocheSolver {
    /// Pd–Ir film constants: `K = 60`, `n = 4`, `Q = 40`, `b = 10`,
    /// `γ_1 = 500`, `γ_2 = 80` (MPa units).
    #[new]
    fn new() -> Self {
        Self {
            k_visco: 60.0,
            n_visco: 4.0,
            q_iso: 40.0,
            b_iso: 10.0,
            gamma1: 500.0,
            gamma2: 80.0,
        }
    }

    /// `σ_y(T) = 220 − 0.2(T − 298.15)` [MPa].
    fn yield_strength(&self, temp_k: f64) -> f64 {
        220.0 - 0.2 * (temp_k - 298.15)
    }

    /// `C_1(T)` [MPa].
    fn c1_modulus(&self, temp_k: f64) -> f64 {
        50.0e3 - 61.54 * (temp_k - 298.15)
    }

    /// `C_2(T)` [MPa].
    fn c2_modulus(&self, temp_k: f64) -> f64 {
        15.0e3 - 21.54 * (temp_k - 298.15)
    }

    /// EPFM leak-before-break evaluation: returns
    /// `(two_a_c_mm, margin, is_compliant)`; mirrors
    /// `shbt_fea_structural::kinetics::evaluate_lbb_margin`.
    #[staticmethod]
    fn evaluate_lbb_margin(
        k_ih: f64,
        sigma_m: f64,
        y_factor: f64,
        a_leak_mm: f64,
    ) -> (f64, f64, bool) {
        let a_c_m = (1.0 / std::f64::consts::PI) * (k_ih / (y_factor * sigma_m)).powi(2);
        let two_a_c_mm = 2.0 * a_c_m * 1000.0;
        let margin = two_a_c_mm / a_leak_mm;
        (two_a_c_mm, margin, margin >= 2.0)
    }

    /// Elastic-shakedown check at cycle 50: `dp_cycle <= 5.15e-8`.
    fn verify_shakedown_state(&self, cycle: usize, dp_cycle: f64) -> PyResult<bool> {
        if cycle >= 50 {
            if dp_cycle > 5.15e-8 {
                return Err(PyValueError::new_err(
                    "Elastic shakedown failure at cycle 50",
                ));
            }
            return Ok(true);
        }
        Ok(false)
    }
}

#[pymodule]
fn shbt_cf_bindings(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTransportConfig>()?;
    module.add_class::<PyQuantumCoherenceSolver>()?;
    module.add_class::<PyChabocheSolver>()?;
    module.add_function(wrap_pyfunction!(solve_mcnabb_foster_transport, module)?)?;
    module.add_function(wrap_pyfunction!(solve_2d_rcwa_identity, module)?)?;
    module.add_function(wrap_pyfunction!(belleville_thermal_compliance, module)?)?;
    Ok(())
}
