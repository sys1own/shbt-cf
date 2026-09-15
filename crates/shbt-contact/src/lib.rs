//! Axisymmetric Belleville washer contact mechanics.
#![allow(missing_docs)]

use ndarray::{Array1, Array2};

pub const REFERENCE_TEMPERATURE_K: f64 = 293.0;

#[derive(Debug, Clone)]
pub struct BellevilleWasherGeometry {
    pub outer_radius: f64,
    pub inner_radius: f64,
    pub thickness: f64,
    pub initial_cone_angle_rad: f64,
    pub thermal_expansion_coeff: f64,
    pub youngs_modulus_293k: f64,
    pub youngs_modulus_temperature_slope: f64,
    pub poissons_ratio: f64,
    pub contact_clearance: f64,
    pub node_count: usize,
}

impl BellevilleWasherGeometry {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.inner_radius <= 0.0 || self.outer_radius <= self.inner_radius {
            return Err("invalid washer radii");
        }
        if self.thickness <= 0.0 || self.node_count < 2 {
            return Err("invalid washer discretization");
        }
        if !(0.0..0.5).contains(&self.poissons_ratio) || self.youngs_modulus_293k <= 0.0 {
            return Err("invalid material properties");
        }
        Ok(())
    }

    pub fn youngs_modulus(&self, temperature_k: f64) -> f64 {
        (self.youngs_modulus_293k
            * (1.0
                - self.youngs_modulus_temperature_slope
                    * (temperature_k - REFERENCE_TEMPERATURE_K)))
            .max(0.01 * self.youngs_modulus_293k)
    }

    pub fn thermal_strain(&self, temperature_k: f64) -> f64 {
        self.thermal_expansion_coeff * (temperature_k - REFERENCE_TEMPERATURE_K)
    }
}

#[derive(Debug, Clone)]
pub struct ActiveContactSet {
    pub active_indices: Vec<usize>,
    pub lagrange_multipliers: Vec<f64>,
    pub displacements: Vec<f64>,
    pub gaps: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

impl ActiveContactSet {
    pub fn is_active(&self, index: usize) -> bool {
        self.active_indices.binary_search(&index).is_ok()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ContactMechanicsSolver {
    pub regularization: f64,
    pub tolerance: f64,
    pub max_iterations: usize,
}

impl Default for ContactMechanicsSolver {
    fn default() -> Self {
        Self {
            regularization: 1.0e8,
            tolerance: 1.0e-8,
            max_iterations: 50,
        }
    }
}

impl ContactMechanicsSolver {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn initialize_pdas_solver(&self, geometry: &BellevilleWasherGeometry) -> ActiveContactSet {
        let nodes = geometry.node_count.max(2);
        ActiveContactSet {
            active_indices: Vec::new(),
            lagrange_multipliers: vec![0.0; nodes],
            displacements: vec![0.0; nodes],
            gaps: vec![geometry.contact_clearance; nodes],
            iterations: 0,
            converged: false,
        }
    }

    /// Solves the scalar radial shell approximation with von Karman membrane stiffening.
    /// The two end nodes are Signorini constraints; interior nodes carry the smooth shell field.
    pub fn solve_step_newton_raphson(
        &self,
        geometry: &BellevilleWasherGeometry,
        active_set: &mut ActiveContactSet,
        temperature_profile: &[f64],
        external_force: f64,
    ) -> Vec<f64> {
        geometry
            .validate()
            .expect("invalid Belleville washer geometry");
        assert!(temperature_profile.len() == geometry.node_count || temperature_profile.len() == 1);
        let temperature = |index: usize| {
            if temperature_profile.len() == 1 {
                temperature_profile[0]
            } else {
                temperature_profile[index]
            }
        };
        let mean_temperature =
            (0..geometry.node_count).map(temperature).sum::<f64>() / geometry.node_count as f64;
        let modulus = geometry.youngs_modulus(mean_temperature);
        let span = geometry.outer_radius - geometry.inner_radius;
        let linear_stiffness = modulus * geometry.thickness.powi(3)
            / (12.0 * (1.0 - geometry.poissons_ratio.powi(2)) * span.powi(2));
        let thermal_displacement = geometry.thermal_strain(mean_temperature)
            * span
            * geometry.initial_cone_angle_rad.cos();
        let membrane_stiffness =
            modulus * geometry.thickness / (1.0 - geometry.poissons_ratio.powi(2));
        let mut displacement = Array1::from_vec(active_set.displacements.clone());
        let mut multipliers = Array1::from_vec(active_set.lagrange_multipliers.clone());
        let mut previous_active = active_set.active_indices.clone();
        let mut previous_displacement = displacement.clone();
        for iteration in 0..self.max_iterations {
            let slope = displacement[geometry.node_count - 1] / span;
            let stiffness = linear_stiffness + membrane_stiffness * slope.powi(2) / span.max(1e-15);
            let free_displacement = (external_force / stiffness) + thermal_displacement;
            for index in 0..geometry.node_count {
                let target = free_displacement * index as f64 / (geometry.node_count - 1) as f64;
                displacement[index] = 0.5 * displacement[index] + 0.5 * target;
            }
            let gaps: Vec<f64> = (0..geometry.node_count)
                .map(|index| geometry.contact_clearance - displacement[index])
                .collect();
            let mut active = Vec::new();
            for &index in &[0, geometry.node_count - 1] {
                multipliers[index] =
                    (external_force.abs() / 2.0 - self.regularization * gaps[index]).max(0.0);
                if multipliers[index] - self.regularization * gaps[index] > 0.0 {
                    active.push(index);
                }
            }
            active.sort_unstable();
            let update = (&displacement - &previous_displacement)
                .mapv(|value| value * value)
                .sum()
                .sqrt();
            let stable = active == previous_active && update < self.tolerance;
            active_set.active_indices = active.clone();
            active_set.gaps = gaps;
            active_set.iterations = iteration + 1;
            active_set.converged = stable;
            if stable {
                break;
            }
            previous_active = active;
            previous_displacement = displacement.clone();
            active_set.displacements = displacement.to_vec();
        }
        active_set.displacements = displacement.to_vec();
        active_set.lagrange_multipliers = multipliers.to_vec();
        active_set.displacements.clone()
    }

    pub fn tangent_stiffness(
        &self,
        geometry: &BellevilleWasherGeometry,
        temperature_k: f64,
        displacement: f64,
    ) -> Array2<f64> {
        let k = geometry.youngs_modulus(temperature_k) * geometry.thickness.powi(3)
            / (12.0
                * (1.0 - geometry.poissons_ratio.powi(2))
                * (geometry.outer_radius - geometry.inner_radius).powi(2));
        Array2::from_diag(&Array1::from_elem(
            geometry.node_count,
            k * (1.0 + displacement.powi(2)),
        ))
    }
}

pub trait ContactMechanics {
    fn initialize_pdas_solver(&self, geometry: &BellevilleWasherGeometry) -> ActiveContactSet;
    fn solve_step_newton_raphson(
        &self,
        geometry: &BellevilleWasherGeometry,
        active_set: &mut ActiveContactSet,
        temperature_profile: &[f64],
        external_force: f64,
    ) -> Vec<f64>;
}

impl ContactMechanics for ContactMechanicsSolver {
    fn initialize_pdas_solver(&self, geometry: &BellevilleWasherGeometry) -> ActiveContactSet {
        self.initialize_pdas_solver(geometry)
    }
    fn solve_step_newton_raphson(
        &self,
        geometry: &BellevilleWasherGeometry,
        active_set: &mut ActiveContactSet,
        temperature_profile: &[f64],
        external_force: f64,
    ) -> Vec<f64> {
        self.solve_step_newton_raphson(geometry, active_set, temperature_profile, external_force)
    }
}
