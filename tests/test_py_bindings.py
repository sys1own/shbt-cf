from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "python"))

from shbt_cf.quantum import CoherenceEngine
from shbt_cf.schemas import load_schema
from shbt_cf.telemetry import ErrorEngine
from shbt_cf.transport import MultiphysicsTransportModel


def test_transport_wrapper_preserves_uniform_state():
    model = MultiphysicsTransportModel.from_parameters(diffusion_coefficient=1.0e-10)
    state = model.solve([2.0] * 8, 3)
    assert np.allclose(state, 2.0)


def test_quantum_wrapper_has_thermal_dephasing_and_filon_rate():
    engine = CoherenceEngine(1.0e6, 2.0)
    assert engine.dephasing_rate(850.0) > engine.dephasing_rate(293.0)
    assert engine.coherence_decay(293.0, [0.0])[0] == 1.0
    assert engine.franck_condon_rate(1.0e-23, 293.0, 1.0e12, 0.5) >= 0.0


def test_gum_monte_carlo_matches_linear_propagation():
    engine = ErrorEngine()
    function = lambda values: [values[0] + values[1]]
    gum = engine.propagate(
        [1.0, 2.0], [[0.01, 0.0], [0.0, 0.04]], function, [[1.0, 1.0]]
    )
    monte_carlo = engine.monte_carlo(
        [1.0, 2.0], [[0.01, 0.0], [0.0, 0.04]], function, samples=5000
    )
    assert engine.compare(gum, monte_carlo, tolerance=0.1)


def test_calibration_schema_resources_load():
    assert (
        load_schema("AnisotropicPermittivitySpectrum")["title"]
        == "AnisotropicPermittivitySpectrum"
    )
    assert load_schema("Pt100CalibrationCurve")["title"] == "Pt100CalibrationCurve"
    assert (
        load_schema("SurfaceRoughnessTopology")["title"] == "SurfaceRoughnessTopology"
    )
