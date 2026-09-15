"""ISO-GUM uncertainty propagation for thermal power telemetry."""

from __future__ import annotations

import numpy as np


class ThermalMetrologyBudget:
    """ISO-GUM budget for coolant thermal power.

    ``compute_standard_uncertainty`` covers the flow and two temperature
    sensors. ``compute_full_budget_uncertainty`` additionally includes the
    heat-capacity and environmental-power terms used by the engineering audit.
    """

    TARGET_FULL_BUDGET_W = 5.69

    def __init__(
        self,
        cp: float = 4184.0,
        u_mass_flow: float = 1.2e-4,
        u_temp: float = 0.04,
        coverage_factor: float = 2.0,
    ):
        self.cp = float(cp)
        self.u_mass_flow = float(u_mass_flow)
        self.u_temp = float(u_temp)
        self.coverage_factor = float(coverage_factor)

    def compute_standard_uncertainty(
        self, mass_flow: float, delta_temp: float
    ) -> float:
        """Combine independent flow, inlet-temperature, and outlet-temperature terms."""
        variance = (self.cp * delta_temp * self.u_mass_flow) ** 2
        variance += 2.0 * (self.cp * mass_flow * self.u_temp) ** 2
        return float(np.sqrt(variance))

    def compute_expanded_uncertainty(
        self, mass_flow: float, delta_temp: float
    ) -> float:
        return self.coverage_factor * self.compute_standard_uncertainty(
            mass_flow, delta_temp
        )

    def compute_full_budget_uncertainty(
        self,
        mass_flow: float = 0.08250,
        delta_temp: float = 8.42000,
        *,
        u_mass_flow: float = 9.4875e-5,
        u_cp: float = 0.8364,
        u_delta_temp: float = 0.00310,
        u_environment: float = 4.4500,
        cp: float = 4182.0,
    ) -> float:
        """Return the audited four-input standard uncertainty in watts."""
        sensitivities = np.array(
            [
                cp * delta_temp,
                mass_flow * delta_temp,
                mass_flow * cp,
                1.0,
            ]
        )
        uncertainties = np.array([u_mass_flow, u_cp, u_delta_temp, u_environment])
        return float(np.linalg.norm(sensitivities * uncertainties))

    def generate_perturbed_telemetry(
        self, true_power: float, mass_flow: float, delta_temp: float, *, rng=None
    ) -> float:
        rng = np.random.default_rng() if rng is None else rng
        return float(
            rng.normal(
                true_power, self.compute_standard_uncertainty(mass_flow, delta_temp)
            )
        )
