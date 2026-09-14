"""Stage-5 tests: NSGA-III, workbench HUD model, and the zero-copy FFI gate."""

from __future__ import annotations

import unittest

import numpy as np

from shbt_cf.native import HAVE_NATIVE
from shbt_cf.optimize import fast_non_dominated_sort, nsga3, reference_points
from shbt_cf.workbench import (
    BellevillePreload,
    GratingAlignment,
    HipimsStress,
    TorqueWrench,
    hud_tick,
    power_net_mc,
    sweep,
)


class TestOptimize(unittest.TestCase):
    def test_reference_points_simplex(self):
        refs = reference_points(2, 4)
        self.assertEqual(refs.shape, (5, 2))
        np.testing.assert_allclose(refs.sum(axis=1), 1.0)

    def test_sort_first_front_dominated(self):
        f = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 5.0]])
        fronts = fast_non_dominated_sort(f)
        np.testing.assert_array_equal(fronts[0], [0, 1, 2])

    def test_zdt1_first_front_spans(self):
        def zdt1(x):
            f1 = x[:, 0]
            g = 1 + 9 * x[:, 1:].mean(1)
            return np.c_[f1, g * (1 - np.sqrt(f1 / g))]

        res = nsga3(zdt1, (np.zeros(4), np.ones(4)), 2, n_pop=60,
                    generations=25, divisions=8, seed=1)
        f1 = res.objectives[res.front, 0]
        self.assertGreater(len(res.front), 10)
        self.assertLess(f1.min(), 0.3)
        self.assertGreater(f1.max(), 0.5)


class TestWorkbench(unittest.TestCase):
    def test_sweep_grid(self):
        rows = sweep(lambda a, b: a + b, {"a": (1.0, 2.0), "b": (10.0, 20.0)})
        self.assertEqual(len(rows), 4)
        self.assertAlmostEqual(rows[3]["value"], 22.0)

    def test_torque_status(self):
        t = TorqueWrench(18.0, 0.5)
        self.assertEqual(t.status(18.2), "ok")
        self.assertEqual(t.status(17.0), "under")
        self.assertEqual(t.status(19.0), "over")

    def test_hud_tick_preload(self):
        hud = hud_tick(
            TorqueWrench(18.0, 0.5),
            BellevillePreload(0.03175, 0.01626, 0.00150, 0.00178, 206e9, 0.30),
            HipimsStress(2.0),
            GratingAlignment(0.4, -0.3, 0.1),
            18.2, 800e-6, -1.4)
        self.assertEqual(hud.torque_status, "ok")
        self.assertEqual(hud.hipims_status, "ok")
        self.assertEqual(hud.alignment_status, "ok")
        self.assertGreater(hud.stack_force_n, 1000.0)


@unittest.skipUnless(HAVE_NATIVE, "shbt_cf_native wheel not built")
class TestNative(unittest.TestCase):
    def test_zerocopy_pointer_identity(self):
        from shbt_cf.native import require_native

        ext = require_native()
        ring = ext.Ring(512)
        view = np.asarray(ring.values_view())
        self.assertEqual(view.ctypes.data, ring.data_ptr())
        self.assertFalse(view.flags["OWNDATA"])
        matrix = np.asarray(ring.frame_matrix())
        self.assertEqual(matrix.ctypes.data, ring.data_ptr())

    def test_power_net_mc_pdf(self):
        pdf = power_net_mc((45.0, 13.25), [[6.25, 0.0], [0.0, 0.81]], n=50_000)
        self.assertAlmostEqual(pdf.mean, 31.75, delta=0.15)
        self.assertAlmostEqual(pdf.std, np.sqrt(6.25 + 0.81), delta=0.15)

    def test_chi2(self):
        from shbt_cf.native import require_native

        ext = require_native()
        self.assertAlmostEqual(ext.chi2([1.0, 2.0], [1.0, 0.25]), 2.0)


if __name__ == "__main__":
    unittest.main()
