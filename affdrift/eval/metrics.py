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


# ── standard generative-model metrics (reported, not judged) ────────────────


def _kth_nn_radius(x: FloatArray, k: int) -> FloatArray:
    """Distance from each point to its k-th nearest neighbour within ``x`` (self excluded)."""
    d = cdist(x, x)
    return np.sort(d, axis=1)[:, k]


def precision_recall(real: FloatArray, fake: FloatArray, k: int = 3) -> tuple[float, float]:
    """Improved precision/recall (Kynkäänniemi et al., 2019) with k-NN manifold estimates."""
    r_real, r_fake = _kth_nn_radius(real, k), _kth_nn_radius(fake, k)
    d = cdist(fake, real)
    precision = float((d <= r_real[None, :]).any(axis=1).mean())
    recall = float((d.T <= r_fake[None, :]).any(axis=1).mean())
    return precision, recall


def density_coverage(real: FloatArray, fake: FloatArray, k: int = 5) -> tuple[float, float]:
    """Density and coverage (Naeem et al., 2020)."""
    r_real = _kth_nn_radius(real, k)
    inside = cdist(fake, real) <= r_real[None, :]
    density = float(inside.sum() / (k * len(fake)))
    coverage = float(inside.any(axis=0).mean())
    return density, coverage


def mmd_rbf(x: FloatArray, y: FloatArray) -> float:
    """Unbiased squared MMD with an RBF kernel, bandwidth by the median heuristic (Gretton et al., 2012)."""
    xy = np.concatenate([x, y])
    d2 = cdist(xy, xy, "sqeuclidean")
    h2 = np.median(d2[np.triu_indices_from(d2, 1)])
    k = np.exp(-d2 / h2)
    n, m = len(x), len(y)
    kxx, kyy, kxy = k[:n, :n], k[n:, n:], k[:n, n:]
    return float((kxx.sum() - n) / (n * (n - 1)) + (kyy.sum() - m) / (m * (m - 1)) - 2 * kxy.mean())


def nearest_mode(samples: FloatArray, centres: FloatArray) -> IntArray:
    """Index of the nearest true mode centre for each sample (mode coverage, plan §5.1)."""
    return cdist(samples, centres).argmin(axis=1)


# ── per-condition evaluation, parallel over conditions ─────────────────────

CONDITION_METRICS = ("w2", "w2_datadata", "mci", "off_manifold", "mode_coverage", "memorization",
                     "precision", "recall", "density", "coverage", "mmd")


def condition_metrics(fake: FloatArray, real: FloatArray, real2: FloatArray, lp_fake: FloatArray, lp_real: FloatArray,
                      centres: FloatArray, train: FloatArray, mem_eps: float) -> dict[str, float]:
    """Every plan §5.1 and standard metric for one condition (pure NumPy/SciPy, safe to run in a worker)."""
    return {
        "w2": w2(fake, real),
        "w2_datadata": w2(real2, real),
        "mci": mci(fake, real),
        "off_manifold": off_manifold(lp_fake, float(np.quantile(lp_real, 0.01))),
        "mode_coverage": mode_coverage(nearest_mode(fake, centres), 3),
        "memorization": memorization(fake, train, mem_eps),
        **dict(zip(("precision", "recall"), precision_recall(real, fake), strict=True)),
        **dict(zip(("density", "coverage"), density_coverage(real, fake), strict=True)),
        "mmd": mmd_rbf(fake, real),
    }


def _chunk(args: tuple) -> list[dict[str, float]]:
    fake, real, real2, lp_fake, lp_real, centres, train, mem_eps = args
    return [condition_metrics(f, r, r2, lf, lr, c, train, mem_eps)
            for f, r, r2, lf, lr, c in zip(fake, real, real2, lp_fake, lp_real, centres, strict=True)]


def evaluate_conditions(fake: FloatArray, real: FloatArray, real2: FloatArray, lp_fake: FloatArray,
                        lp_real: FloatArray, centres: FloatArray, train: FloatArray, mem_eps: float,
                        *, workers: int = 5) -> dict[str, list[float]]:
    """:func:`condition_metrics` for every condition, split over ``workers`` processes (1 = in this process)."""
    import concurrent.futures

    n = len(fake)
    bounds = np.linspace(0, n, max(1, workers) * 3 + 1).astype(int)
    jobs = [(fake[a:b], real[a:b], real2[a:b], lp_fake[a:b], lp_real[a:b], centres[a:b], train, mem_eps)
            for a, b in zip(bounds, bounds[1:]) if b > a]
    if workers <= 1:
        rows = [row for job in jobs for row in _chunk(job)]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            rows = [row for part in pool.map(_chunk, jobs) for row in part]
    return {k: [row[k] for row in rows] for k in CONDITION_METRICS}
