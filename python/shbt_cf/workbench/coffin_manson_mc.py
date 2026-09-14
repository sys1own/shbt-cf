"""Physically bounded Coffin-Manson fatigue sampling."""

from __future__ import annotations

import numpy as np
from scipy.stats import truncnorm


def compute_fatigue_lifetime_mc(
    epsilon_f_prime: float,
    c: float,
    mean_strain: float,
    std_dev_strain: float,
    iterations: int,
    *,
    min_strain: float = 1.0e-6,
    max_strain: float = 0.45,
    seed: int | None = None,
) -> np.ndarray:
    """Return fatigue cycles for bounded plastic-strain Monte Carlo draws."""
    if epsilon_f_prime <= 0 or c == 0:
        raise ValueError("epsilon_f_prime must be positive and c must be non-zero")
    if std_dev_strain <= 0 or iterations < 1:
        raise ValueError("std_dev_strain must be positive and iterations must be positive")
    if not 0 < min_strain < max_strain:
        raise ValueError("strain bounds must satisfy 0 < min_strain < max_strain")

    alpha = (min_strain - mean_strain) / std_dev_strain
    beta = (max_strain - mean_strain) / std_dev_strain
    sampler = truncnorm(alpha, beta, loc=mean_strain, scale=std_dev_strain)
    strain = sampler.rvs(size=iterations, random_state=seed)
    return 0.5 * (epsilon_f_prime / strain) ** (1.0 / c)
