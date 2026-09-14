//! Finite spatial-temporal Fourier-Bloch elastodynamic eigenproblem.

/// Periodic scalarized Fourier-Bloch model for a selected elastic polarization.
#[derive(Clone, Debug, PartialEq)]
pub struct FourierBlochModel {
    /// Bloch wavevector [1/m].
    pub wavevector: f64,
    /// Reciprocal lattice vectors [1/m].
    pub reciprocal_vectors: Vec<f64>,
    /// Temporal harmonics.
    pub harmonics: Vec<i32>,
    /// Modulation frequency [rad/s].
    pub modulation_frequency: f64,
    /// Mass density [kg/m³].
    pub density: f64,
    /// Fourier coefficients of the scalar stiffness field, indexed by harmonic difference.
    pub stiffness_coefficients: Vec<f64>,
}

impl FourierBlochModel {
    fn coefficient(&self, difference: i32) -> f64 {
        let centre = (self.stiffness_coefficients.len() / 2) as i32;
        self.stiffness_coefficients
            .get((difference + centre) as usize)
            .copied()
            .unwrap_or(0.0)
    }

    /// Assembles the finite Galerkin matrix of the space-time action.
    pub fn matrix(&self) -> Vec<Vec<f64>> {
        let n = self.reciprocal_vectors.len() * self.harmonics.len();
        let mut matrix = vec![vec![0.0; n]; n];
        for (row, &g) in self.reciprocal_vectors.iter().enumerate() {
            for (hrow, &harmonic) in self.harmonics.iter().enumerate() {
                let i = row * self.harmonics.len() + hrow;
                let q = self.wavevector + g;
                matrix[i][i] =
                    q * q / self.density + (harmonic as f64 * self.modulation_frequency).powi(2);
                for (column, &g_prime) in self.reciprocal_vectors.iter().enumerate() {
                    for (hcolumn, &harmonic_prime) in self.harmonics.iter().enumerate() {
                        let j = column * self.harmonics.len() + hcolumn;
                        matrix[i][j] += (self.wavevector + g)
                            * (self.wavevector + g_prime)
                            * self.coefficient(harmonic - harmonic_prime)
                            / self.density;
                    }
                }
            }
        }
        matrix
    }

    /// Computes eigenvalues of the real symmetric Galerkin matrix by Jacobi rotations.
    pub fn eigenvalues(&self) -> Vec<f64> {
        let mut matrix = self.matrix();
        let n = matrix.len();
        for _ in 0..(32 * n.max(1)) {
            if n < 2 {
                break;
            }
            let mut p = 0;
            let mut q = 1;
            for i in 0..n {
                for j in (i + 1)..n {
                    if matrix[i][j].abs() > matrix[p][q].abs() {
                        p = i;
                        q = j;
                    }
                }
            }
            if matrix[p][q].abs() < 1e-13 {
                break;
            }
            let angle = 0.5 * (2.0 * matrix[p][q]).atan2(matrix[q][q] - matrix[p][p]);
            let (sin, cos) = angle.sin_cos();
            for row in matrix.iter_mut().take(n) {
                let ip = row[p];
                let iq = row[q];
                row[p] = cos * ip - sin * iq;
                row[q] = sin * ip + cos * iq;
            }
            let (before, after) = matrix.split_at_mut(q);
            let row_p = &mut before[p];
            let row_q = &mut after[0];
            for (value_p, value_q) in row_p.iter_mut().zip(row_q.iter_mut()).take(n) {
                let pi = *value_p;
                let qi = *value_q;
                *value_p = cos * pi - sin * qi;
                *value_q = sin * pi + cos * qi;
            }
        }
        let mut values: Vec<_> = (0..n).map(|i| matrix[i][i]).collect();
        values.sort_by(f64::total_cmp);
        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uncoupled_bloch_modes_are_real_and_sorted() {
        let model = FourierBlochModel {
            wavevector: 2.0,
            reciprocal_vectors: vec![0.0, 1.0],
            harmonics: vec![0],
            modulation_frequency: 0.0,
            density: 2.0,
            stiffness_coefficients: vec![0.0, 3.0, 0.0],
        };
        let values = model.eigenvalues();
        assert_eq!(values.len(), 2);
        assert!(values[0].is_finite() && values[0] <= values[1]);
    }
}
