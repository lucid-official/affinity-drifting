"""Metrics.  Level 0 (assignment, TV, moments) and Level 1 (W2, MCI, ...).

Pure NumPy on host copies: metrics are computed once per run, after the
dynamics, so precision beats speed here.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from affdrift._typing import FloatArray, IntArray

# ── Level 0 ─────────────────────────────────────────────────────────────────


def nearest(points: FloatArray, centres: FloatArray) -> IntArray:
    """Index of the nearest centre for every point; ``points`` is ``(..., D)``."""
    flat = points.reshape(-1, points.shape[-1])
    return cdist(flat, centres).argmin(axis=1).reshape(points.shape[:-1])


def assignment_matrix(particles: FloatArray, y: FloatArray) -> FloatArray:
    """``A[k, j]``: share of condition k's particles whose nearest positive is ``y_j``."""
    idx = nearest(particles, y)
    k = len(y)
    return np.stack([np.bincount(row, minlength=k) / row.size for row in idx])


def tv_rows(p: FloatArray, q: FloatArray) -> FloatArray:
    """Total variation between matching rows of two row-stochastic matrices."""
    return 0.5 * np.abs(p - q).sum(axis=-1)


def own_distance(particles: FloatArray, y: FloatArray) -> FloatArray:
    """Mean distance of each condition's particles to its own positive ``y_k``."""
    return np.linalg.norm(particles - y[:, None, :], axis=-1).mean(axis=1)


def moments(points: FloatArray) -> tuple[FloatArray, float]:
    """Sample mean and mean per-dimension variance (ddof = 0)."""
    return points.mean(axis=0), float(points.var(axis=0).mean())


# ── Level 1 ─────────────────────────────────────────────────────────────────


def w2(a: FloatArray, b: FloatArray) -> float:
    """Exact W2 between two equal-size empirical measures (optimal assignment)."""
    if len(a) != len(b):
        raise ValueError("w2 needs equal sample sizes")
    cost = cdist(a, b, "sqeuclidean")
    rows, cols = linear_sum_assignment(cost)
    return float(np.sqrt(cost[rows, cols].mean()))


def mci(q: FloatArray, p: FloatArray) -> float:
    """Mean-collapse index 1 − tr Cov_q / tr Cov_p (1 = fully collapsed)."""
    return 1.0 - float(np.trace(np.cov(q, rowvar=False)) / np.trace(np.cov(p, rowvar=False)))


def off_manifold(log_density: FloatArray, threshold: float) -> float:
    """Share of samples whose true log-density is below ``threshold``."""
    return float((log_density < threshold).mean())


def mode_coverage(mode_index: IntArray, n_modes: int, min_share: float = 0.05) -> float:
    """Share of true modes that receive at least ``min_share`` of the samples."""
    shares = np.bincount(mode_index, minlength=n_modes) / mode_index.size
    return float((shares >= min_share).mean())


def memorization(samples: FloatArray, train: FloatArray, eps: float) -> float:
    """Share of samples within ``eps`` of their nearest training point."""
    return float((cdist(samples, train).min(axis=1) <= eps).mean())
