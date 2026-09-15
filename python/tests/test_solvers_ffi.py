"""ctypes FFI tests for the update-10.1 solver exports.

Skipped automatically when the cdylibs have not been built
(``cargo build -p shbt-rcwa`` / ``cargo build`` at the workspace root).
"""

from __future__ import annotations

import unittest

import numpy as np

from shbt_cf.solvers import (
    NativeSolverUnavailable,
    _find_lib,
    run_coupled_simulation,
    solve_floquet,
)

_HAVE_RCWA = _find_lib("libshbt_rcwa") is not None
_HAVE_CF = _find_lib("libshbt_cf") is not None


@unittest.skipUnless(_HAVE_RCWA, "libshbt_rcwa cdylib not built")
class TestFloquetLuFfi(unittest.TestCase):
    def test_known_3x3_solve(self):
        a = np.array(
            [
                [4.0 + 1.0j, -2.0 + 0.5j, 1.0 - 1.0j],
                [0.0 + 2.0j, 3.0 + 0.0j, -1.0 + 1.0j],
                [2.0 - 0.5j, 1.0 + 1.0j, 5.0 - 2.0j],
            ]
        )
        b = np.array([1.0, 1.0j, -1.0 + 2.0j])
        x = solve_floquet(a, b)
        np.testing.assert_allclose(a @ x, b, atol=1e-10)

    def test_illconditioned_residual(self):
        n = 4
        h = 1.0 / (np.arange(n)[:, None] + np.arange(n)[None, :] + 1.0)
        a = h * (1.0 + 0.5j)
        b = np.array([1.0 - 0.5j, 0.25 + 0.75j, -2.0, 0.5 + 0.5j])
        x = solve_floquet(a, b)
        residual = np.max(np.abs(a @ x - b))
        self.assertLess(residual, 1e-9)

    def test_singular_matrix_raises(self):
        a = np.ones((2, 2), dtype=np.complex128)
        with self.assertRaises(RuntimeError):
            solve_floquet(a, np.ones(2, dtype=np.complex128))

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            solve_floquet(np.eye(2, dtype=np.complex128), np.ones(3))


@unittest.skipUnless(_HAVE_CF, "libshbt_cf cdylib not built")
class TestCoupledSimulationFfi(unittest.TestCase):
    def test_bop_master_power_table(self):
        result = run_coupled_simulation(
            steps=1000,
            dt=1e-3,
            lindblad={
                "omega_0": 0.8,
                "omega_l": 1.3,
                "delta": 0.2,
                "g_nuc": 0.5,
                "det_nuc": -0.1,
                "gamma_1": 0.05,
                "gamma_2": 0.02,
            },
            bop={
                "q_flow": 2.0833e-4,
                "delta_p": 85_000.0,
                "eta_pump": 0.65,
                "q_lattice": 147.66,
                "cop": 3.0,
            },
        )
        self.assertAlmostEqual(result["p_pump"], 27.2432, places=3)
        self.assertAlmostEqual(result["q_total"], 174.9032, places=3)
        self.assertAlmostEqual(result["p_compressor"], 58.3011, places=3)
        np.testing.assert_allclose(np.trace(result["rho"]), 1.0, atol=1e-9)

    def test_missing_library_raises(self):
        if _HAVE_CF:
            self.skipTest("library present")
        with self.assertRaises(NativeSolverUnavailable):
            run_coupled_simulation(1, 1e-3, {}, {})


if __name__ == "__main__":
    unittest.main()
