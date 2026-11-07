r"""Per-condition caption bandwidth from the effective-neighbour rule.

:math:`N_\mathrm{eff}(c) = (\sum_i w_i)^2 / \sum_i w_i^2` grows from 1 at
:math:`h \to 0` to the bank size at :math:`h \to \infty`.  B0 picks, for each
condition, the smallest :math:`h_p(c)` with :math:`N_\mathrm{eff}(c) \ge N^*`
(plan §2.1), found by bisection in :math:`\log h`.
"""

import numpy as np

from affdrift._typing import FloatArray
from affdrift.core.kernels import caption_log_weights


def n_eff(log_w: FloatArray) -> FloatArray:
    """Effective number of neighbours of each row of (possibly unnormalised) log-weights."""
    log_w = np.atleast_2d(log_w)
    w = np.exp(log_w - log_w.max(axis=-1, keepdims=True))
    return w.sum(axis=-1) ** 2 / (w**2).sum(axis=-1)


def solve_hp(
    psi_q: FloatArray,
    psi_bank: FloatArray,
    n_star: float,
    *,
    rtol: float = 1e-10,
    max_iter: int = 200,
) -> FloatArray:
    r"""Smallest :math:`h` per query row with :math:`N_\mathrm{eff} \ge N^*`.

    Raises
    ------
    ValueError
        If ``n_star`` exceeds the bank size, which no bandwidth can reach.
    """
    psi_q = np.atleast_2d(np.asarray(psi_q, dtype=np.float64))
    psi_bank = np.atleast_2d(np.asarray(psi_bank, dtype=np.float64))
    n_bank = len(psi_bank)
    if n_star > n_bank:
        raise ValueError(f"N* = {n_star} exceeds the bank size {n_bank}")
    if n_star <= 1:
        return np.zeros(len(psi_q))

    d = np.sqrt(((psi_q[:, None, :] - psi_bank[None, :, :]) ** 2).sum(axis=-1))
    positive = d[d > 0]
    lo = np.full(len(psi_q), np.log(positive.min() * 1e-3))
    hi = np.full(len(psi_q), np.log(d.max() * 1e3))

    def reached(log_h: FloatArray) -> np.ndarray:
        return n_eff(caption_log_weights(psi_q, psi_bank, np.exp(log_h))) >= n_star * (1 - 1e-12)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        ok = reached(mid)
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid)
        if np.all(hi - lo < rtol):
            break
    return np.exp(hi)
