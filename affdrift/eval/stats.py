"""Seed statistics: percentile bootstrap CIs, paired across arms (CLAUDE.md §2.10)."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

N_RESAMPLE = 10_000
LEVEL = 0.95


@dataclass(frozen=True, slots=True)
class CI:
    mean: float
    lo: float
    hi: float
    n: int

    def within(self, lo: float, hi: float) -> bool:
        """The whole interval lies in ``[lo, hi]``."""
        return lo <= self.lo and self.hi <= hi

    def outside(self, lo: float, hi: float) -> bool:
        """The whole interval lies outside ``[lo, hi]``."""
        return self.hi < lo or self.lo > hi

    def __str__(self) -> str:
        return f"{self.mean:.4g} [{self.lo:.4g}, {self.hi:.4g}]"

    def to_json(self) -> dict[str, float | int]:
        return asdict(self)


def bootstrap_ci(values: Sequence[float], *, n: int = N_RESAMPLE, level: float = LEVEL, seed: int = 0) -> CI:
    """Percentile bootstrap CI of the mean."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        raise ValueError("no values")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n, x.size))].mean(axis=1)
    alpha = (1 - level) / 2
    return CI(float(x.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha)), x.size)


def paired_bootstrap_ci(a: Sequence[float], b: Sequence[float], **kwargs: float) -> CI:
    """CI of mean(a − b) over seeds that both arms share, resampled as pairs."""
    a_, b_ = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a_.shape != b_.shape:
        raise ValueError("paired bootstrap needs the same seeds in both arms")
    return bootstrap_ci(a_ - b_, **kwargs)
