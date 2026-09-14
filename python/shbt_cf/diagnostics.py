"""Traceable calorimetry, PSD, and helium-4 diagnostic calculations."""

from __future__ import annotations

import math

from scipy.stats import beta


class ShbtDiagnosticProcessor:
    def __init__(self, r_nominal: float = 50.0000, u_r: float = 0.00035,
                 u_v_relative: float = 1.2e-5) -> None:
        self.r_nominal = r_nominal
        self.u_r = u_r
        self.u_v_relative = u_v_relative

    def calculate_gum_calorimetry(self, target_powers):
        if self.r_nominal <= 0 or self.u_r < 0 or self.u_v_relative < 0:
            raise ValueError("calibration parameters must be non-negative and resistance positive")
        results = []
        for power in target_powers:
            if not math.isfinite(power) or power < 0:
                raise ValueError("target powers must be finite and non-negative")
            voltage = math.sqrt(power * self.r_nominal)
            u_voltage = voltage * self.u_v_relative
            c_voltage = 2.0 * voltage / self.r_nominal
            c_resistance = -(voltage ** 2) / (self.r_nominal ** 2)
            standard = math.hypot(c_voltage * u_voltage, c_resistance * self.u_r)
            expanded = 2.0 * standard
            results.append({"Target_Power_W": power, "Voltage_V": voltage,
                            "Std_Uncertainty_W": standard,
                            "Expanded_Uncertainty_W": expanded,
                            "Relative_Uncertainty_Pct": expanded / power * 100.0 if power else 0.0})
        return results

    def process_psd_validation(self, total_trials: int, observed_leakage: int,
                               confidence: float = 0.95):
        if total_trials <= 0 or not 0 <= observed_leakage <= total_trials:
            raise ValueError("leakage count must be within positive trial count")
        if not 0 < confidence < 1:
            raise ValueError("confidence must be between zero and one")
        upper = float(beta.ppf(confidence, observed_leakage + 1,
                               total_trials - observed_leakage))
        return upper, 1.0 / upper if upper > 0 else math.inf

    def model_helium_quantitation(self, heat_power_kw: float, operational_days: float,
                                  background_leak_rate: float):
        if heat_power_kw < 0 or operational_days < 0 or background_leak_rate < 0:
            raise ValueError("helium inputs must be non-negative")
        moles = 3.756e-5 * heat_power_kw * operational_days
        mass_g = moles * 4.002602
        purity = 1.0 if moles == 0 and background_leak_rate == 0 else (
            0.0 if moles == 0 else max(0.0, 1.0 - background_leak_rate / (moles * 1.45e5)))
        return moles, mass_g, purity