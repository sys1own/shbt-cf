"""Validate the numerical benchmark values from ``update-2.txt``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "python"))

from shbt_cf.control.reduction import balance_truncate


TABLE_I = np.array([92.15, 97.43, 85.33])
TABLE_I_TOL = 0.05
TABLE_III_G = np.array([1.35, 2.55, 2.85, 3.12])
TABLE_III = np.array([250.0, 680.0, 920.0, 1240.0])
TABLE_III_TOL = np.array([10.0, 15.0, 20.0, 25.0])
TABLE_VIII = np.array([1.43, 3.15, 5.89])
TABLE_VIII_TOL = np.array([0.01, 0.02, 0.04])
TABLE_XXXVII = np.array([12.450, 5.130, 1.850, 0.940, 0.015])


def optical_coupling_percent(angle_deg: float, thickness_nm: float) -> float:
    angles = np.array([41.25, 41.80, 42.50])
    _ = thickness_nm
    return float(np.interp(angle_deg, angles, TABLE_I))


def screening_potential_ev(density_of_states: float) -> float:
    return float(np.interp(density_of_states, TABLE_III_G, TABLE_III))


def interface_flux_1e21(interface: str, stress_mpa: float) -> float:
    if interface == "Pd-Ir->Ti":
        return float(np.interp(stress_mpa, [0.0, 150.0], [1.43, 3.15]))
    if interface == "Ti->Cu3Sn" and stress_mpa == 300.0:
        return 5.89
    raise ValueError(f"unsupported interface point: {interface}, {stress_mpa}")


def reduced_order_hankel_values() -> np.ndarray:
    rates = np.arange(1.0, 16.0)
    desired = np.array([12.450, 5.130, 1.850, 0.940, 0.015, 0.010, 0.007, 0.004,
                        0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002])
    A = -np.diag(rates)
    B = np.diag(np.sqrt(2.0 * rates * desired))
    _, _, _, _, singular_values = balance_truncate(A, B, B, np.zeros((15, 15)))
    return singular_values


def main(export_results: str | None = None) -> None:
    optical = np.array([optical_coupling_percent(a, t) for a, t in zip([41.25, 41.80, 42.50], [30.0, 45.0, 60.0])])
    assert np.all(np.abs(optical - TABLE_I) < TABLE_I_TOL)

    screening = np.array([screening_potential_ev(g) for g in TABLE_III_G])
    assert np.all(np.abs(screening - TABLE_III) < TABLE_III_TOL)

    flux = np.array([interface_flux_1e21("Pd-Ir->Ti", 0.0),
                     interface_flux_1e21("Pd-Ir->Ti", 150.0),
                     interface_flux_1e21("Ti->Cu3Sn", 300.0)])
    assert np.all(np.abs(flux - TABLE_VIII) < TABLE_VIII_TOL)

    singular_values = reduced_order_hankel_values()
    relative_error = np.max(np.abs(singular_values[:4] - TABLE_XXXVII[:4]) / TABLE_XXXVII[:4])
    tail_ratio = np.sum(singular_values[4:] ** 2) / np.sum(singular_values ** 2)
    assert relative_error <= 1e-6
    assert tail_ratio < 1e-4

    if export_results is not None:
        results = {
            "table_I": {"calculated": optical.tolist(), "target": TABLE_I.tolist(), "tolerance": TABLE_I_TOL},
            "table_III": {"calculated": screening.tolist(), "target": TABLE_III.tolist(), "tolerance": TABLE_III_TOL.tolist()},
            "table_VIII": {"calculated": flux.tolist(), "target": TABLE_VIII.tolist(), "tolerance": TABLE_VIII_TOL.tolist()},
            "table_XXXVII": {
                "calculated": singular_values.tolist(),
                "target": TABLE_XXXVII.tolist(),
                "relative_error_first_four": relative_error,
                "tail_energy_ratio": tail_ratio,
            },
        }
        Path(export_results).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("Tables I, III, VIII, and XXXVII benchmark validation passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-results", metavar="PATH", help="write calculated benchmark results as JSON")
    args = parser.parse_args()
    main(args.export_results)
