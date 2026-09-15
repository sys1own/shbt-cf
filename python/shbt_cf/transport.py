"""Python orchestration for the native McNabb-Foster transport solver."""

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
class MultiphysicsTransportModel:
    """Validated wrapper for the native Crank-Nicolson transport kernel."""

    config: object | None = None

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = _bindings().PyTransportConfig()

    @classmethod
    def from_parameters(cls, **parameters: float) -> "MultiphysicsTransportModel":
        return cls(_bindings().PyTransportConfig(**parameters))

    def solve(
        self,
        lattice_initial: Sequence[float],
        steps: int,
        *,
        dt: float = 1.0e-3,
        temperature: float = 293.0,
        interface_boundaries: Sequence[int] = (),
        segregation_factors: Sequence[float] | None = None,
    ) -> list[float]:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        return list(
            _bindings().solve_mcnabb_foster_transport(
                self.config,
                list(lattice_initial),
                steps,
                dt,
                temperature,
                list(interface_boundaries),
                None if segregation_factors is None else list(segregation_factors),
            )
        )

    def step(
        self,
        lattice_state: Sequence[float],
        *,
        dt: float = 1.0e-3,
        temperature: float = 293.0,
    ) -> list[float]:
        return self.solve(lattice_state, 1, dt=dt, temperature=temperature)
