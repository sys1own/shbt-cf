//! Stable large-order multi-phonon rate evaluation.

/// Update-8 phonon order for the 23.84 MeV transition.
pub const PHONON_ORDER: f64 = 6.92e8;

/// Logarithm of the uniform large-order approximation to `I_N(z)`.
/// Working in log space prevents overflow for the resonant argument.
pub fn log_modified_bessel_uniform(order: f64, argument: f64) -> f64 {
    assert!(order > 0.0 && argument > 0.0);
    let root = (order * order + argument * argument).sqrt();
    -0.5 * (2.0 * std::f64::consts::PI).ln() - 0.25 * (order * order + argument * argument).ln()
        + root
        + order * (argument / (order + root)).ln()
}

/// Critical multi-phonon rate using the algebraic resonant asymptote [s^-1].
pub fn critical_lattice_rate(
    coherent_cells: f64,
    single_cell_matrix_element_ev: f64,
    phonon_bandwidth_hz: f64,
    order: f64,
) -> f64 {
    assert!(coherent_cells > 0.0 && single_cell_matrix_element_ev > 0.0);
    assert!(phonon_bandwidth_hz > 0.0 && order > 0.0);
    let hbar_ev_s: f64 = 6.582_119_569e-16;
    let matrix_element_ev = coherent_cells * single_cell_matrix_element_ev;
    matrix_element_ev.powi(2)
        / (hbar_ev_s.powi(2) * phonon_bandwidth_hz * (2.0 * std::f64::consts::PI * order).sqrt())
}

/// Computes the lattice branching fraction without subtractive cancellation.
pub fn lattice_branching_fraction(gamma_lattice: f64, gamma_gamma: f64) -> f64 {
    assert!(gamma_lattice >= 0.0 && gamma_gamma >= 0.0);
    if gamma_lattice == 0.0 {
        0.0
    } else {
        1.0 / (1.0 + gamma_gamma / gamma_lattice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resonant_rate_dominates_radiative_channel() {
        let rate = critical_lattice_rate(1.0e12, 10.0, 1.0e13, PHONON_ORDER);
        assert!(rate > 1.0e20, "rate = {rate:e}");
        assert!(lattice_branching_fraction(rate, 1.0e14) > 0.999999);
    }

    #[test]
    fn uniform_bessel_log_is_finite_at_order_scale() {
        assert!(log_modified_bessel_uniform(PHONON_ORDER, PHONON_ORDER).is_finite());
    }
}
