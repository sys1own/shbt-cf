from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "python"))

from shbt_cf.diagnostics import ShbtDiagnosticProcessor
from shbt_cf.fea_automation import generate_shbt_apdl_script
from shbt_cf.native import require_native


def test_pyo3_update5_kernel():
    kernel = require_native().ShbtNumericalKernel()
    assert kernel.precision_bits == 512
    assert kernel.verify_state_reduction([12.45, 5.13, 1.85, 0.94]) is True
    high_precision, binary64, drift = kernel.compute_floquet_inversion(350.0, -298.57)
    assert high_precision.startswith("3.672507641671")
    assert abs(binary64 - 3.672507641671456) < 1e-12
    assert float(drift) >= 0.0


def test_gum_outputs_and_psd_bounds():
    processor = ShbtDiagnosticProcessor()
    result = processor.calculate_gum_calorimetry([200.0])[0]
    assert result["Voltage_V"] == (200.0 * 50.0) ** 0.5
    assert result["Expanded_Uncertainty_W"] > result["Std_Uncertainty_W"]
    upper, rejection = processor.process_psd_validation(2_995_731, 0)
    assert 0.0 < upper < 1e-5
    assert rejection > 1e6


def test_apdl_contains_all_target_layers():
    script = generate_shbt_apdl_script()
    assert "ET,1,186" in script
    assert script.count("BLOCK,") == 5
    assert "TUNIF,350.0" in script


def test_native_domain_simulation_is_live():
    values = require_native().run_domain_simulation()
    assert values[1] == pytest.approx(8.32806399587, rel=1e-12)
    assert values[5] == pytest.approx(2.53e14)
