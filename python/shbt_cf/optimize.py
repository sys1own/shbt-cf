"""NSGA-III multi-objective optimizer (Deb & Jain 2014): non-dominated
sorting + Das-Dennis reference-direction niching, SBX crossover and
polynomial mutation. Pure NumPy, deterministic given a seed.
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

ObjectiveFn = Callable[[np.ndarray], np.ndarray]  # (n_pop, d) -> (n_pop, m)

# First-principles mechanical limits for the active 20 mm core interface.
D_CORE = 0.020
A_CONTACT = np.pi * D_CORE**2 / 4.0
F_MIN_LIMIT = 6283.19
F_MAX_LIMIT = 78539.82

# Reconciled thermomechanical design values.
K_STACK_OPTIMIZED = 5.0e6
DL_NET_TARGET = 20.1e-6
EPSILON_P_CEILING = 0.0004805


def apply_physical_constraints(force_array: np.ndarray | float) -> np.ndarray | float:
    """Clip clamp force values to the vacuum-sealing/yield envelope."""
    return np.clip(force_array, F_MIN_LIMIT, F_MAX_LIMIT)


def compute_structural_penalty(force: np.ndarray | float, temperature: float = 350.0):
    """Penalize Pd-Ir yield and minimum sealing-pressure violations."""
    if temperature >= 350.0:
        yield_strength_film = 250.0e6
    else:
        t_fraction = (temperature - 25.0) / 325.0
        yield_strength_film = (350.0 - t_fraction * 100.0) * 1.0e6

    contact_stress = np.asarray(force) / A_CONTACT
    penalty = np.zeros_like(contact_stress, dtype=float)
    stress_violation = np.maximum(contact_stress - yield_strength_film, 0.0)
    pressure_violation = np.maximum(20.0e6 - contact_stress, 0.0)
    penalty += 1e5 * (stress_violation / 1e6) ** 2
    penalty += 1e5 * (pressure_violation / 1e6) ** 2
    if np.ndim(force) == 0:
        return float(penalty)
    return penalty


def reference_points(m: int, divisions: int) -> np.ndarray:
    """Das-Dennis simplex-lattice reference points for `m` objectives."""
    out = []

    def rec(rem: int, pos: int, acc: list[int]) -> None:
        if pos == m - 1:
            acc.append(rem)
            out.append(np.array(acc, dtype=float) / divisions)
            acc.pop()
            return
        for v in range(rem + 1):
            acc.append(v)
            rec(rem - v, pos + 1, acc)
            acc.pop()

    rec(divisions, 0, [])
    return np.asarray(out)


def fast_non_dominated_sort(f: np.ndarray) -> list[np.ndarray]:
    """Minimisation fronts; returns index arrays per rank (vectorised)."""
    # dom[i, j] is True iff j dominates i.
    le = np.all(f[None, :, :] <= f[:, None, :], axis=2)
    lt = np.any(f[None, :, :] < f[:, None, :], axis=2)
    dom = le & lt
    dom_count = dom.sum(axis=1)
    fronts: list[np.ndarray] = []
    active = np.ones(len(f), dtype=bool)
    while True:
        front = np.where(active & (dom_count == 0))[0]
        if front.size == 0:
            break
        fronts.append(front)
        active[front] = False
        dom_count -= dom[:, front].sum(axis=1)
    return fronts


@dataclass
class Nsga3Result:
    """Pareto approximation."""

    population: np.ndarray  # decision vectors
    objectives: np.ndarray  # (n, m)
    front: np.ndarray  # indices of first-front members


def _niching(front_len: int, f_norm: np.ndarray, refs: np.ndarray, need: int) -> list[int]:
    """NSGA-III niching over the last front: returns positions into it."""
    chosen: list[int] = []
    # perpendicular distance to each reference line
    def assoc(idx: int) -> tuple[int, float]:
        f = f_norm[idx]
        # distance to line through ref direction
        with np.errstate(divide="ignore", invalid="ignore"):
            proj = (f @ refs.T) / np.maximum((refs * refs).sum(axis=1), 1e-14)
            d = np.linalg.norm(f - np.outer(proj, np.ones(refs.shape[1])) * refs, axis=1)
        r = int(np.argmin(d))
        return r, float(d[r])

    ref_count = np.zeros(len(refs), dtype=int)
    remaining = list(range(front_len))
    pool = {i: assoc(i) for i in remaining}
    # first pass: one member per sparsest reference
    while len(chosen) < need and remaining:
        counts = {r: ref_count[r] for r in {pool[i][0] for i in remaining}}
        r_star = min(counts, key=counts.get)
        cand = [i for i in remaining if pool[i][0] == r_star]
        i = min(cand, key=lambda j: pool[j][1])
        chosen.append(i)
        remaining.remove(i)
        ref_count[r_star] += 1
    return chosen


def nsga3(
    objective: ObjectiveFn,
    bounds: tuple[np.ndarray, np.ndarray],
    n_objectives: int,
    n_pop: int = 92,
    generations: int = 40,
    divisions: int = 12,
    seed: int = 1,
) -> Nsga3Result:
    """Minimise `objective` (shape `(n, m)`) over box bounds via NSGA-III."""
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    d = lo.size
    rng = np.random.default_rng(seed)
    refs = reference_points(n_objectives, divisions)
    pop = lo + (hi - lo) * rng.random((n_pop, d))
    eta_c, eta_m, p_mut = 20.0, 20.0, 1.0 / d

    def mate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        u = rng.random(a.shape)
        beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)), (1 / (2 * (1 - u))) ** (1 / (eta_c + 1)))
        child = 0.5 * ((1 + beta) * a + (1 - beta) * b)
        mut = rng.random(a.shape) < p_mut
        delta = np.where(rng.random(a.shape) < 0.5, (2 * rng.random(a.shape)) ** (1 / (eta_m + 1)) - 1, 1 - (2 * (1 - rng.random(a.shape))) ** (1 / (eta_m + 1)))
        child = np.where(mut, child + delta * (hi - lo), child)
        return np.clip(child, lo, hi)

    for _ in range(generations):
        i = rng.permutation(n_pop)
        kids = np.array([mate(pop[i[k]], pop[i[k + 1]]) for k in range(0, n_pop - n_pop % 2, 2)])
        comb = np.vstack([pop, kids])
        f = objective(comb)
        fronts = fast_non_dominated_sort(f)
        keep: list[int] = []
        for fr in fronts:
            if len(keep) + len(fr) <= n_pop:
                keep.extend(fr)
            else:
                need = n_pop - len(keep)
                ideal = f[keep + list(fr)].min(axis=0)
                nadir = f[keep + list(fr)].max(axis=0)
                f_norm = (f[fr] - ideal) / np.maximum(nadir - ideal, 1e-14)
                keep.extend(fr[j] for j in _niching(len(fr), f_norm, refs, need))
                break
        pop = comb[keep]

    f = objective(pop)
    fronts = fast_non_dominated_sort(f)
    return Nsga3Result(population=pop, objectives=f, front=fronts[0])


def _run_benchmark_export(path: str | Path) -> None:
    """Validate the published benchmark tables and emit the simulator JSON artifact."""
    script = Path(__file__).resolve().parents[2] / "tests" / "validate_benchmarks.py"
    spec = importlib.util.spec_from_file_location("validate_benchmarks", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load benchmark validator from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    module.main(str(out))


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line interface for optimizer validation and artifact export."""
    parser = argparse.ArgumentParser(description="SHBT optimization and benchmark export")
    parser.add_argument(
        "--export-results",
        action="store_true",
        help="validate the benchmark tables and write a simulator_output.json artifact",
    )
    parser.add_argument(
        "--out",
        default="simulator_output.json",
        help="output path for the generated JSON artifact",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.export_results:
        _run_benchmark_export(args.out)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
