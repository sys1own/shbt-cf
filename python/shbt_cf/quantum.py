"""Python wrapper for coherence and Franck-Condon calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def _bindings():
    try:
        import shbt_cf_bindings as bindings
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "shbt_cf_bindings is not built. Run: maturin develop "
            "-m crates/shbt-py-bindings/Cargo.toml"
        ) from exc
    return bindings


@dataclass
class CoherenceEngine:
    """Manage temperature-scaled dephasing and thermal phase variance."""

    larmor_frequency: float
    coupling_constant: float
    base_dephasing_rate: float = 1.0e-4
    reference_temperature: float = 293.0

    def __post_init__(self) -> None:
        self._solver = _bindings().PyQuantumCoherenceSolver(
            self.larmor_frequency,
            self.coupling_constant,
            self.base_dephasing_rate,
            self.reference_temperature,
        )

    def dephasing_rate(self, temperature: float) -> float:
        return float(self._solver.dephasing_rate(temperature))

    def coherence_decay(
        self, temperature: float, time_steps: Sequence[float]
    ) -> list[float]:
        return list(self._solver.solve_coherence_decay(temperature, list(time_steps)))

    def thermal_variance(self, temperature: float) -> float:
        return float(self._solver.thermal_variance(temperature))

    def franck_condon_rate(
        self,
        energy: float,
        temperature: float,
        phonon_frequency: float,
        huang_rhys: float,
        *,
        periods: int = 8,
        samples_per_period: int = 64,
    ) -> float:
        return float(
            self._solver.franck_condon_rate(
                energy,
                temperature,
                phonon_frequency,
                huang_rhys,
                periods,
                samples_per_period,
            )
        )
