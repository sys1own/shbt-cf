"""End-to-end verification of the update-9 native multiphysics path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "python"))

import shbt_cf_bindings as native

from shbt_cf.quantum import CoherenceEngine
from shbt_cf.schemas import SCHEMA_DIR
from shbt_cf.telemetry import ErrorEngine
from shbt_cf.transport import MultiphysicsTransportModel


def test_full_multiphysics_pipeline_and_calibration_schemas():
    mode_count, reflection, transmission = native.solve_2d_rcwa_identity(
        25, 25, 960.8e-9
    )
    assert mode_count == 625
    assert reflection == pytest.approx(1.0)
    assert transmission == pytest.approx(1.0)

    deflection, stiffness = native.belleville_thermal_compliance(
        0.02, 0.01, 0.0005, 200e9, 0.3, 850.0, 12e-6, 293.0, 10.0
    )
    assert stiffness > 0.0
    assert deflection > 0.0

    transport = MultiphysicsTransportModel.from_parameters(diffusion_coefficient=1e-10)
    transported = transport.solve(
        [1.0] * 16,
        4,
        dt=1e-3,
        temperature=850.0,
        interface_boundaries=[8],
        segregation_factors=[1.0],
    )
    assert np.all(np.isfinite(transported))
    assert np.allclose(transported, 1.0)

    coherence = CoherenceEngine(1.0e6, 2.0)
    decay = coherence.coherence_decay(850.0, [0.0, 1.0, 2.0])
    assert decay[0] == pytest.approx(1.0)
    assert decay[0] > decay[1] > decay[2] > 0.0

    error_engine = ErrorEngine()
    mass_flow, heat_capacity, delta_temperature = 0.0825, 4182.0, 8.42
    voltage, current = 48.0, 12.0
    means = [mass_flow, delta_temperature, heat_capacity, voltage, current]
    covariance = np.diag([9.4875e-5**2, 0.0031**2, 0.8364**2, 0.04**2, 0.02**2])

    def calorimetry(values: np.ndarray) -> list[float]:
        return [values[0] * values[1] * values[2], values[3] * values[4]]

    gum = error_engine.propagate(
        means,
        covariance,
        calorimetry,
        error_engine.calorimetry_jacobian(*means),
    )
    monte_carlo = error_engine.monte_carlo(
        means, covariance, calorimetry, samples=20_000
    )
    assert gum.mean[0] == pytest.approx(mass_flow * heat_capacity * delta_temperature)
    assert error_engine.compare(gum, monte_carlo, tolerance=0.08)

    examples = {
        "AnisotropicPermittivitySpectrum": {
            "material_identifier": "Pd-Ir",
            "spectral_points": [
                {
                    "wavelength_nm": 785.0,
                    "epsilon_xx": [1.0, 0.0],
                    "epsilon_yy": [1.0, 0.0],
                    "epsilon_zz": [1.0, 0.0],
                }
            ],
        },
        "Pt100CalibrationCurve": {
            "calibration_id": "PT100-UPDATE9",
            "coefficients": {"A": 3.9083e-3, "B": -5.775e-7, "C": 0.0},
            "uncertainty_rto": 0.1,
        },
        "SurfaceRoughnessTopology": {
            "surface_id": "interface-01",
            "rq_nm": 2.0,
            "autocorrelation_length_nm": 40.0,
            "profile_matrix": [[0.0, 1.0], [1.0, 0.0]],
        },
    }
    for schema_name, instance in examples.items():
        schema_path = SCHEMA_DIR / f"{schema_name}.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.validate(instance, schema)
        jsonschema.validate(yaml.safe_load(yaml.safe_dump(instance)), schema)
