r"""Level 1: the conditional 3-mode GMM in :math:`\mathbb R^2` (plan §5.1, CLAUDE.md §5).

For :math:`c = (c_1, c_2) \in [0, 1]^2` and :math:`j = 0, 1, 2`:

* centre angle :math:`2\pi c_1 + 2\pi j / 3`, radius :math:`1 + c_2`;
* mode standard deviation ``mode_std`` (0.1);
* mixture weights :math:`\mathrm{softmax}_j\,\beta (c_2 - \tfrac12)(j - 1)`.

The plan says only "mixture 비율은 c2에 따라 softmax로 변화"; the logit form
and :math:`\beta = 2` are this module's choice, recorded in
docs/deviations.md (D9) to be confirmed before Level 1.
"""

import numpy as np
from scipy.special import log_softmax, logsumexp

from affdrift._typing import FloatArray, IntArray

N_MODES = 3


def mode_params(c: FloatArray, *, mix_beta: float) -> tuple[FloatArray, FloatArray]:
    """``(N, 3, 2)`` centres and ``(N, 3)`` log mixture weights."""
    c = np.atleast_2d(c)
    j = np.arange(N_MODES)
    angle = 2 * np.pi * c[:, :1] + 2 * np.pi * j / N_MODES
    radius = 1.0 + c[:, 1:2]
    centres = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=-1)
    log_pi = log_softmax(mix_beta * (c[:, 1:2] - 0.5) * (j - 1), axis=-1)
    return centres, log_pi


def sample_conditions(n: int, rng: np.random.Generator) -> FloatArray:
    return rng.uniform(0.0, 1.0, size=(n, 2))


def sample_x(c: FloatArray, n: int, rng: np.random.Generator, *, mode_std: float, mix_beta: float
             ) -> tuple[FloatArray, IntArray]:
    """``(N, n, 2)`` samples of p(x|c) and their ``(N, n)`` mode indices."""
    centres, log_pi = mode_params(c, mix_beta=mix_beta)
    u = rng.uniform(size=(len(centres), n, 1))
    modes = (u > np.cumsum(np.exp(log_pi), axis=-1)[:, None, :]).sum(-1).clip(max=N_MODES - 1)
    x = np.take_along_axis(centres, modes[..., None], axis=1) + mode_std * rng.normal(size=(len(centres), n, 2))
    return x, modes


def log_density(x: FloatArray, c: FloatArray, *, mode_std: float, mix_beta: float) -> FloatArray:
    """``(N, n)`` log p(x|c) for ``x`` of shape ``(N, n, 2)``."""
    centres, log_pi = mode_params(c, mix_beta=mix_beta)
    d2 = ((x[:, :, None, :] - centres[:, None, :, :]) ** 2).sum(-1)
    log_n = -d2 / (2 * mode_std**2) - np.log(2 * np.pi * mode_std**2)
    return logsumexp(log_n + log_pi[:, None, :], axis=-1)


def conditional_mean(c: FloatArray, *, mix_beta: float) -> FloatArray:
    """E[x|c] — where single positive collapses to (plan §5.1, C1)."""
    centres, log_pi = mode_params(c, mix_beta=mix_beta)
    return (np.exp(log_pi)[..., None] * centres).sum(axis=1)
