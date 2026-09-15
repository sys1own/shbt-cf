"""ISO/IEC Guide 98-3 first-order GUM propagation and Monte Carlo checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class GumResult:
    mean: np.ndarray
    covariance: np.ndarray
    standard_uncertainty: np.ndarray


@dataclass(frozen=True)
class MonteCarloResult:
    mean: np.ndarray
    covariance: np.ndarray
    standard_uncertainty: np.ndarray
    samples: np.ndarray


class ErrorEngine:
    """Propagate correlated measurement uncertainty and verify it by sampling."""

    def propagate(
        self,
        means: Sequence[float],
        covariance: Sequence[Sequence[float]],
        function: Callable[[np.ndarray], Sequence[float]],
        jacobian: Sequence[Sequence[float]] | None = None,
    ) -> GumResult:
        mean_vector = np.asarray(means, dtype=float)
        covariance_matrix = np.asarray(covariance, dtype=float)
        if covariance_matrix.shape != (mean_vector.size, mean_vector.size):
            raise ValueError("covariance must be square and match means")
        output_mean = np.asarray(function(mean_vector), dtype=float)
        jacobian_matrix = (
            self._finite_difference_jacobian(function, mean_vector)
            if jacobian is None
            else np.asarray(jacobian, dtype=float)
        )
        output_covariance = jacobian_matrix @ covariance_matrix @ jacobian_matrix.T
        return GumResult(
            output_mean,
            output_covariance,
            np.sqrt(np.maximum(np.diag(output_covariance), 0.0)),
        )

    def monte_carlo(
        self,
        means: Sequence[float],
        covariance: Sequence[Sequence[float]],
        function: Callable[[np.ndarray], Sequence[float]],
        *,
        samples: int = 100_000,
        seed: int = 0x5EED,
    ) -> MonteCarloResult:
        if samples <= 1:
            raise ValueError("samples must be greater than one")
        mean_vector = np.asarray(means, dtype=float)
        draws = np.random.default_rng(seed).multivariate_normal(
            mean_vector, np.asarray(covariance, dtype=float), samples
        )
        outputs = np.asarray([function(draw) for draw in draws], dtype=float)
        covariance_result = np.atleast_2d(np.cov(outputs, rowvar=False, ddof=1))
        return MonteCarloResult(
            outputs.mean(axis=0),
            covariance_result,
            outputs.std(axis=0, ddof=1),
            outputs,
        )

    def compare(
        self, gum: GumResult, monte_carlo: MonteCarloResult, tolerance: float = 0.02
    ) -> bool:
        relative_error = np.abs(
            gum.standard_uncertainty - monte_carlo.standard_uncertainty
        ) / np.maximum(monte_carlo.standard_uncertainty, 1e-30)
        return bool(np.all(relative_error <= tolerance))

    @staticmethod
    def _finite_difference_jacobian(
        function: Callable[[np.ndarray], Sequence[float]], point: np.ndarray
    ) -> np.ndarray:
        baseline = np.asarray(function(point), dtype=float)
        result = np.empty((baseline.size, point.size), dtype=float)
        for index in range(point.size):
            step = np.sqrt(np.finfo(float).eps) * max(abs(point[index]), 1.0)
            plus = point.copy()
            minus = point.copy()
            plus[index] += step
            minus[index] -= step
            result[:, index] = (
                np.asarray(function(plus)) - np.asarray(function(minus))
            ) / (2.0 * step)
        return result

    @staticmethod
    def calorimetry_jacobian(
        mass_flow: float,
        delta_temperature: float,
        heat_capacity: float,
        voltage: float,
        current: float,
    ) -> np.ndarray:
        return np.array(
            [
                [
                    heat_capacity * delta_temperature,
                    mass_flow * heat_capacity,
                    mass_flow * delta_temperature,
                    0.0,
                    0.0,
                ],
                [0.0, 0.0, 0.0, current, voltage],
            ]
        )
