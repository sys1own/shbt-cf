"""ctypes FFI wrappers for the C-ABI solver exports (update-10.1 §4).

Loads the workspace cdylibs directly — no maturin/PyO3 build required:

* ``libshbt_rcwa.so``  → ``shbt_rcwa_solve_floquet`` (complex LU solve A·x = b)
* ``libshbt_cf.so``    → ``shbt_cf_run_coupled_simulation`` (Lindblad RK4 + BOP)

The libraries are produced by ``cargo build`` (``cdylib`` crate-type) and are
resolved from ``$SHBT_CF_TARGET_DIR`` or the repo's ``target/{release,debug}``.
All functions raise :class:`NativeSolverUnavailable` when the cdylib has not
been built yet so callers can degrade to pure-Python fallbacks.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np


class NativeSolverUnavailable(RuntimeError):
    """Raised when the Rust cdylib is not built on this host."""


def _find_lib(name: str) -> Optional[Path]:
    names = [f"{name}.so", f"{name}.dylib", f"{name}.dll"]
    search = []
    env_dir = os.environ.get("SHBT_CF_TARGET_DIR")
    if env_dir:
        search.append(Path(env_dir))
    root = Path(__file__).resolve().parents[2]
    search += [root / "target" / "release", root / "target" / "debug"]
    for base in search:
        for fname in names:
            candidate = base / fname
            if candidate.exists():
                return candidate
    return None


def _load(name: str) -> ctypes.CDLL:
    path = _find_lib(name)
    if path is None:
        raise NativeSolverUnavailable(
            f"{name} not built; run `cargo build -p {name.replace('lib', '')} "
            f"(cdylib)` or set SHBT_CF_TARGET_DIR"
        )
    return ctypes.CDLL(str(path))


def solve_floquet(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve the complex ``A x = b`` Floquet system via ``shbt_rcwa_solve_floquet``.

    ``a`` must be an ``(n, n)`` complex matrix, ``b`` a length-``n`` complex
    vector. Returns the complex solution ``x``. Raises ``RuntimeError`` on a
    singular matrix and ``ValueError`` on shape mismatches.
    """
    a = np.ascontiguousarray(a, dtype=np.complex128)
    b = np.ascontiguousarray(b, dtype=np.complex128)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("a must be a square matrix")
    n = a.shape[0]
    if b.shape != (n,):
        raise ValueError("b must be a length-n vector")

    lib = _load("libshbt_rcwa")
    dbl = ctypes.POINTER(ctypes.c_double)
    lib.shbt_rcwa_solve_floquet.restype = ctypes.c_int32
    lib.shbt_rcwa_solve_floquet.argtypes = [dbl, dbl, dbl, dbl, ctypes.c_size_t, dbl, dbl]

    # The C ABI wants two contiguous f64 planes; `a.real`/`.imag` are strided
    # views, so materialize contiguous copies for both inputs and outputs.
    a_re = np.ascontiguousarray(a.real, dtype=np.float64)
    a_im = np.ascontiguousarray(a.imag, dtype=np.float64)
    b_re = np.ascontiguousarray(b.real, dtype=np.float64)
    b_im = np.ascontiguousarray(b.imag, dtype=np.float64)
    x_re = np.zeros(n, dtype=np.float64)
    x_im = np.zeros(n, dtype=np.float64)
    status = lib.shbt_rcwa_solve_floquet(
        a_re.ctypes.data_as(dbl),
        a_im.ctypes.data_as(dbl),
        b_re.ctypes.data_as(dbl),
        b_im.ctypes.data_as(dbl),
        ctypes.c_size_t(n),
        x_re.ctypes.data_as(dbl),
        x_im.ctypes.data_as(dbl),
    )
    x = x_re + 1j * x_im
    if status == -2:
        raise RuntimeError("shbt_rcwa_solve_floquet: singular matrix")
    if status != 0:
        raise RuntimeError(f"shbt_rcwa_solve_floquet failed with status {status}")
    return x


def run_coupled_simulation(
    steps: int,
    dt: float,
    lindblad: Dict[str, float],
    bop: Dict[str, float],
) -> Dict[str, object]:
    """Run the Lindblad RK4 + balance-of-plant simulation via the C ABI.

    ``lindblad`` keys: ``omega_0``, ``omega_l``, ``delta``, ``g_nuc``,
    ``det_nuc``, ``gamma_1``, ``gamma_2``.
    ``bop`` keys: ``q_flow``, ``delta_p``, ``eta_pump``, ``q_lattice``, ``cop``.

    Returns ``{"rho": (3, 3) complex ndarray, "p_pump": float,
    "q_total": float, "p_compressor": float}``.
    """
    lib = _load("libshbt_cf")
    f = lib.shbt_cf_run_coupled_simulation
    f.restype = ctypes.c_int32
    dbl = ctypes.POINTER(ctypes.c_double)
    f.argtypes = [ctypes.c_size_t] + [ctypes.c_double] * 13 + [dbl] * 5

    rho_re = np.zeros(9, dtype=np.float64)
    rho_im = np.zeros(9, dtype=np.float64)
    out = (ctypes.c_double * 3)()
    status = f(
        ctypes.c_size_t(steps),
        dt,
        lindblad["omega_0"],
        lindblad["omega_l"],
        lindblad["delta"],
        lindblad["g_nuc"],
        lindblad["det_nuc"],
        lindblad["gamma_1"],
        lindblad["gamma_2"],
        bop["q_flow"],
        bop["delta_p"],
        bop["eta_pump"],
        bop["q_lattice"],
        bop["cop"],
        rho_re.ctypes.data_as(dbl),
        rho_im.ctypes.data_as(dbl),
        ctypes.cast(ctypes.byref(out, 0), dbl),
        ctypes.cast(ctypes.byref(out, 8), dbl),
        ctypes.cast(ctypes.byref(out, 16), dbl),
    )
    if status != 0:
        raise RuntimeError(f"shbt_cf_run_coupled_simulation failed with status {status}")
    return {
        "rho": (rho_re + 1j * rho_im).reshape(3, 3),
        "p_pump": out[0],
        "q_total": out[1],
        "p_compressor": out[2],
    }
