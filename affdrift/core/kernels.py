r"""The two kernels of AffD: :math:`k_x` on samples and :math:`w_i(c)` on conditions.

``k_x`` is evaluated every step on device, in Lucid.  The caption weights
depend only on the frozen :math:`\psi` and the bank, so they are computed
once, on the host, in float64, and handed to the field as log-weights.

Every log-weight returned here is normalised per query row
(:math:`\log\sum_i w_i = 0`), so a weight set is a distribution over the bank.
"""

from enum import StrEnum

import lucid
import numpy as np
from lucid import Tensor
from scipy.special import logsumexp

from affdrift._typing import FloatArray


class XKernel(StrEnum):
    """Form of :math:`k_x`.  ``NONE`` is control K1 (:math:`k_x \\equiv 1`)."""

    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"
    NONE = "none"


def sqdist(x: Tensor, y: Tensor) -> Tensor:
    """``(..., N, D) × (..., M, D) → (..., N, M)`` squared Euclidean distances.

    Expanded-square form, clamped at zero against cancellation.
    """
    xx = (x * x).sum(dim=-1).unsqueeze(-1)
    yy = (y * y).sum(dim=-1).unsqueeze(-2)
    return lucid.clamp(xx + yy - 2.0 * lucid.matmul(x, y.mT), min=0.0)


def log_x_kernel(d2: Tensor, *, kind: XKernel, sigma: float) -> Tensor:
    r""":math:`\log k_x` from squared distances.

    Gaussian :math:`-d^2/2\sigma^2`, Laplace :math:`-d/\sigma` (the explicit
    product form of E2.5), or zero for K1.
    """
    match kind:
        case XKernel.GAUSSIAN:
            return d2 * (-0.5 / sigma**2)
        case XKernel.LAPLACE:
            return lucid.sqrt(d2 + 1e-12) * (-1.0 / sigma)
        case XKernel.NONE:
            return d2 * 0.0


def log_normalize(logits: FloatArray) -> FloatArray:
    """Subtract each row's log-sum-exp so the weights of a row sum to one."""
    return logits - logsumexp(logits, axis=-1, keepdims=True)


def caption_log_weights(psi_q: FloatArray, psi_bank: FloatArray, h: FloatArray | float) -> FloatArray:
    r"""Normalised :math:`\log w_i(c)`, :math:`w_i(c) = \exp(-\lVert\psi(c)-\psi(c_i)\rVert^2 / 2h^2)`.

    ``h`` is a scalar or one bandwidth per query row.  The two limits are
    exact rather than numerical, so single positive and K5 run the same code:

    * ``h = 0`` — one-hot on the nearest bank entry (all of them on a tie):
      the single-positive arm.
    * ``h = inf`` — uniform over the bank: control K5.
    """
    psi_q = np.atleast_2d(np.asarray(psi_q, dtype=np.float64))
    psi_bank = np.atleast_2d(np.asarray(psi_bank, dtype=np.float64))
    d2 = ((psi_q[:, None, :] - psi_bank[None, :, :]) ** 2).sum(axis=-1)
    h = np.broadcast_to(np.asarray(h, dtype=np.float64), (len(psi_q),))
    if np.any(h < 0) or np.any(np.isnan(h)):
        raise ValueError("bandwidths must be non-negative")

    logits = np.empty_like(d2)
    zero, inf = h == 0, np.isinf(h)
    mid = ~(zero | inf)
    logits[mid] = -d2[mid] / (2.0 * h[mid, None] ** 2)
    logits[inf] = 0.0
    nearest = d2[zero] <= d2[zero].min(axis=1, keepdims=True)
    logits[zero] = np.where(nearest, 0.0, -np.inf)
    return log_normalize(logits)


def shuffle_neighbors(log_w: FloatArray, rng: np.random.Generator) -> FloatArray:
    """Control K2: each row's weights on a random permutation of the bank.

    The multiset of weights in a row is unchanged, so N_eff is exactly the
    same as the semantic weights it replaces; only *which* neighbours are
    borrowed becomes random.
    """
    return np.stack([row[rng.permutation(row.shape[0])] for row in log_w])
