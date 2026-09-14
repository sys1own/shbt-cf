"""Balanced truncation and Lur'e-Postnikov stability certification."""

from __future__ import annotations

import numpy as np
import scipy.linalg as la


def balance_truncate(A, B, C, D, target_dim: int = 4):
    """Return a balanced, truncated realization and its Hankel singular values."""
    A, B, C, D = map(np.asarray, (A, B, C, D))
    n = A.shape[0]
    if A.shape != (n, n) or B.shape[0] != n or C.shape[1] != n:
        raise ValueError("incompatible state-space dimensions")
    if not 1 <= target_dim <= n:
        raise ValueError("target_dim must be between one and the state dimension")
    if np.max(np.real(la.eigvals(A))) >= 0:
        raise ValueError("balanced truncation requires a Hurwitz A matrix")

    Wc = la.solve_continuous_lyapunov(A, -(B @ B.T))
    Wo = la.solve_continuous_lyapunov(A.T, -(C.T @ C))
    Wc = (Wc + Wc.T) / 2.0
    Wo = (Wo + Wo.T) / 2.0
    ec, Lc = la.eigh(Wc)
    eo, Lo = la.eigh(Wo)
    if ec.min() <= 0 or eo.min() <= 0:
        raise ValueError("controllability and observability Gramians must be positive definite")
    Lc = Lc @ np.diag(np.sqrt(ec))
    Lo = Lo @ np.diag(np.sqrt(eo))
    U, singular_values, Vh = la.svd(Lo.T @ Lc, full_matrices=False)
    singular_values = np.maximum(singular_values, 0.0)
    keep = singular_values > np.finfo(float).eps
    if target_dim > int(keep.sum()):
        raise ValueError("target_dim exceeds the numerical Hankel rank")
    inv_sqrt = np.diag(1.0 / np.sqrt(singular_values[keep]))
    T = Lc @ Vh.T[:, keep] @ inv_sqrt
    T_inv = inv_sqrt @ U.T[keep] @ Lo.T
    T_r, T_inv_r = T[:, :target_dim], T_inv[:target_dim, :]
    return (T_inv_r @ A @ T_r, T_inv_r @ B, C @ T_r, D.copy(), singular_values)


def verify_lure_stability(Ar, Br, Cr, gamma, *, solver: str = "SCS") -> bool:
    """Certify the sector-bounded reduced model with a CVXPY LMI."""
    import cvxpy as cp

    Ar, Br, Cr = map(np.asarray, (Ar, Br, Cr))
    gamma = np.asarray(gamma, dtype=float).reshape(-1)
    n, q = Ar.shape[0], Br.shape[1]
    if Br.shape != (n, q) or Cr.shape[1] != n or gamma.size != q or np.any(gamma <= 0):
        raise ValueError("incompatible Lur'e dimensions or non-positive sector")
    P = cp.Variable((n, n), symmetric=True)
    lam = cp.Variable(q, nonneg=True)
    Lambda = cp.diag(lam)
    top = Ar.T @ P + P @ Ar
    off = P @ Br - Cr.T @ Lambda
    bottom = -2.0 * Lambda @ np.diag(1.0 / gamma)
    lmi = cp.bmat([[top, off], [off.T, bottom]])
    eps = 1e-7
    problem = cp.Problem(cp.Minimize(0), [P >> eps * np.eye(n), lam >= eps, lmi << -eps * np.eye(n + q)])
    try:
        problem.solve(solver=solver, verbose=False)
    except cp.error.SolverError:
        problem.solve(solver="CLARABEL", verbose=False)
    return problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
