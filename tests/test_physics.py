from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1] / "python"))

from shbt_cf.control.reduction import balance_truncate
from shbt_cf.native import require_native


def test_coordinate_metric_invariance():
    coordinates = np.array([[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]])
    angle = 0.371
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                         [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    transformed = coordinates @ rotation.T
    assert np.max(np.abs(np.linalg.norm(transformed, axis=1) - np.linalg.norm(coordinates, axis=1))) < 1e-12


def test_gauss_kronrod_polynomial_reference():
    solver = require_native().ConformalKineticsSolver(512)
    calculated = solver.integrate_polynomial([0.0, 0.0, 1.0], 0.0, 1.0)
    assert abs(calculated - 1.0 / 3.0) < 1e-15


def test_mcnabb_foster_constant_state_has_zero_mass_drift():
    native = require_native()
    size = 32
    solver = native.McNabbFosterSolver(
        1.0, [1.0] * size, [1.0] * size, [1.0] * size, [2.0] * size,
        [0.0] * size, [[] for _ in range(size)], [[] for _ in range(size)], [[] for _ in range(size)],
        300.0, [],
    )
    initial_mass = size * 2.0
    solver.step(0.01)
    final_mass = sum(solver.concentrations())
    assert abs(final_mass - initial_mass) / initial_mass < 1e-12


def test_hankel_tail_is_small_for_reference_model():
    rates = np.arange(1.0, 16.0)
    A = -np.diag(rates)
    B = np.diag(np.exp(-rates / 2.0))
    C = B.copy()
    _, _, _, _, singular_values = balance_truncate(A, B, C, np.zeros((15, 15)))
    tail = np.sum(singular_values[4:] ** 2) / np.sum(singular_values ** 2)
    assert tail < 1e-4
