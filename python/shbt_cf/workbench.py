"""Dual-mode operational workbench (Section 5 of simulator_spec.pdf).

* Offline Theoretical Closure Mode — parameter sweeps, NSGA-III
  optimisation drivers and GUM Monte-Carlo PDFs over the native
  metrology engine.
* Physical Fabrication Workbench Mode — cleanroom HUD state model:
  digital torque wrench feedback, Belleville stack preload, HiPIMS
  stress warnings and RCWA grating alignment errors.

All numerics run through the `shbt_cf_native` extension; a pure-Python
DIN 2092 fallback keeps the HUD usable before the wheel is built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from shbt_cf.native import native, require_native
from shbt_cf.optimize import Nsga3Result, nsga3


# ---------------------------------------------------------------------------
# Offline Theoretical Closure Mode


def sweep(fn, grid: dict[str, tuple[float, ...]]) -> list[dict[str, float]]:
    """Evaluate `fn(**point)` over a Cartesian product grid of parameters."""
    keys = list(grid)
    rows = []
    for combo in product(*(grid[k] for k in keys)):
        point = dict(zip(keys, combo))
        point["value"] = float(fn(**point))
        rows.append(point)
    return rows


def optimize(objective, bounds, n_objectives: int, **kwargs) -> Nsga3Result:
    """Run NSGA-III over `objective` — thin wrapper for the workbench."""
    return nsga3(objective, bounds, n_objectives, **kwargs)


@dataclass
class MonteCarloPdf:
    """Histogram/PDF summary of a GUM Supplement-1 propagation run."""

    samples: np.ndarray
    centers: np.ndarray
    density: np.ndarray
    mean: float
    std: float
    coverage_95: tuple[float, float]


def power_net_mc(mean: tuple[float, float], cov, n: int = 1_000_000, seed: int = 7,
                 threads: int = 4) -> MonteCarloPdf:
    """P_net = P_teg - P_support Monte-Carlo PDF (native engine)."""
    ext = require_native()
    mean_out, std, lo95, hi95, samples = ext.power_net_mc(
        list(mean), [list(r) for r in cov], n, seed, threads)
    return _pdf(np.asarray(samples), (lo95, hi95))


def coffin_manson_mc(delta_eps_p=(0.0004805, 0.000024025), eps_f=(0.35, 0.02),
                     c=(0.58, 0.01), n: int = 1_000_000, seed: int = 7,
                     threads: int = 4) -> MonteCarloPdf:
    """Coffin-Manson cycle-life PDF (native engine)."""
    ext = require_native()
    _median, lo95, hi95, samples = ext.coffin_manson_mc(eps_f, c, delta_eps_p, n, seed, threads)
    return _pdf(np.asarray(samples), (lo95, hi95))


def _pdf(samples: np.ndarray, coverage_95: tuple[float, float], bins: int = 120) -> MonteCarloPdf:
    samples = samples[np.isfinite(samples)]
    hist, edges = np.histogram(samples, bins=bins, density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return MonteCarloPdf(samples, centers, hist, float(samples.mean()),
                         float(samples.std()), coverage_95)


# ---------------------------------------------------------------------------
# Physical Fabrication Workbench Mode


@dataclass
class TorqueWrench:
    """Digital torque wrench channel (cleanroom fastener tightening)."""

    setpoint_nm: float
    tolerance_nm: float

    def status(self, reading_nm: float) -> str:
        err = reading_nm - self.setpoint_nm
        if abs(err) <= self.tolerance_nm:
            return "ok"
        return "under" if err < 0 else "over"


@dataclass
class BellevillePreload:
    """Belleville stack preload state (SI units; DIN 2092 force model)."""

    de_m: float
    di_m: float
    t_m: float
    h0_m: float
    e_pa: float
    nu: float

    def force_per_disc_n(self, deflection_m: float) -> float:
        if native is not None:
            return float(native.disc_force(self.de_m, self.di_m, self.t_m,
                                           self.h0_m, deflection_m, self.e_pa, self.nu))
        return _disc_force_py(self.de_m, self.di_m, self.t_m, self.h0_m,
                              deflection_m, self.e_pa, self.nu)

    def stack_force_n(self, deflection_m: float, n_parallel: int = 1) -> float:
        return n_parallel * self.force_per_disc_n(deflection_m)


def _disc_force_py(de, di, t, h0, s, e, nu):
    """DIN 2092 ideal-disc axial force; mirrors `DiscSpring::force`."""
    delta = de / di
    ln = np.log(delta)
    k1 = ((delta - 1) / delta) ** 2 / (np.pi * ((delta + 1) / (delta - 1) - 2 / ln))
    scale = 4 * e / (1 - nu**2) * t**4 / (k1 * de**2)
    x, h = s / t, h0 / t
    return scale * x * ((h - x) * (h - 0.5 * x) + 1)


@dataclass
class HipimsStress:
    """HiPIMS film-stress monitor: warns when residual stress nears limits."""

    limit_gpa: float

    def status(self, stress_gpa: float) -> str:
        if abs(stress_gpa) > self.limit_gpa:
            return "exceeds-limit"
        if abs(stress_gpa) > 0.8 * self.limit_gpa:
            return "warn"
        return "ok"


@dataclass
class GratingAlignment:
    """RCWA grating alignment error in in-plane shift and tilt."""

    dx_um: float
    dy_um: float
    theta_tilt_mrad: float
    tol_xy_um: float = 2.0
    tol_tilt_mrad: float = 0.5

    def status(self) -> str:
        if abs(self.dx_um) > self.tol_xy_um or abs(self.dy_um) > self.tol_xy_um:
            return "shift-out"
        if abs(self.theta_tilt_mrad) > self.tol_tilt_mrad:
            return "tilt-out"
        return "ok"


@dataclass
class HudState:
    """One tick of the fabrication HUD."""

    torque_nm: float
    torque_status: str
    stack_deflection_m: float
    stack_force_n: float
    hipims_stress_gpa: float
    hipims_status: str
    alignment: GratingAlignment
    alignment_status: str
    residuals: dict[str, float] = field(default_factory=dict)


def hud_tick(torque: TorqueWrench, preload: BellevillePreload,
             stress: HipimsStress, alignment: GratingAlignment,
             torque_nm: float, deflection_m: float, stress_gpa: float,
             residuals: dict[str, float] | None = None) -> HudState:
    """Assemble one HUD tick from live sensor readings."""
    return HudState(
        torque_nm=torque_nm,
        torque_status=torque.status(torque_nm),
        stack_deflection_m=deflection_m,
        stack_force_n=preload.stack_force_n(deflection_m),
        hipims_stress_gpa=stress_gpa,
        hipims_status=stress.status(stress_gpa),
        alignment=alignment,
        alignment_status=alignment.status(),
        residuals=residuals or {},
    )


CHANNEL_IDS = {"thermocouple_n": 1, "fbg_strain": 2, "ccd": 3}


def ring_residual_sample(capacity: int = 4096, frames: int = 2000,
                         dt_ns: int = 100_000) -> dict[str, float]:
    """Push synthetic 10 kHz telemetry through the native SPSC ring and score
    each channel's chi-squared residual against a zero-offset surrogate."""
    ext = require_native()
    ring = ext.Ring(capacity)
    rng = np.random.default_rng(11)
    truth = {"thermocouple_n": 300.0, "fbg_strain": 0.0, "ccd": 0.5}
    sigma = {"thermocouple_n": 0.25, "fbg_strain": 5e-4, "ccd": 2e-3}
    chi2: dict[str, list[float]] = {k: [] for k in truth}
    for i in range(frames):
        for name, mu in truth.items():
            meas = mu + sigma[name] * rng.standard_normal()
            ring.push_wave(CHANNEL_IDS[name], i, 1, meas, dt_ns)
            chi2[name].append(float(ext.chi2([meas - mu], [1.0 / sigma[name]**2])))
    return {k: float(np.mean(v)) for k, v in chi2.items()}
