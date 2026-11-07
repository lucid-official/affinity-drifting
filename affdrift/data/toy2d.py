r"""Level 1: the conditional 3-mode GMM in :math:`\mathbb R^2` (plan §5.1, CLAUDE.md §5).

For :math:`c = (c_1, c_2) \in [0, 1]^2` and :math:`j = 0, 1, 2`:

* centre angle :math:`2\pi c_1 + 2\pi j / 3`, radius :math:`1 + c_2`;
* mode standard deviation ``mode_std`` (0.1);
* mixture weights :math:`\mathrm{softmax}_j\,\beta (c_2 - \tfrac12)(j - 1)`.

The plan says only "mixture 비율은 c2에 따라 softmax로 변화"; the logit form
and :math:`\beta = 2` are this module's choice, recorded in
docs/deviations.md (D9, confirmed as D18).
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import log_softmax, logsumexp

from affdrift._typing import FloatArray, IntArray
from affdrift.runtime import numpy_rng

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


@dataclass(frozen=True, slots=True)
class Bank:
    """The L1 data: N unique conditions with one positive each, plus the fixed evaluation sets.

    Attributes
    ----------
    c, y
        ``(N, 2)`` training conditions and their single positives.
    eval_train
        Indices into the bank of the training conditions used for evaluation.
    c_heldout
        ``(H, 2)`` held-out conditions (never in the bank).
    """

    c: FloatArray
    y: FloatArray
    eval_train: IntArray
    c_heldout: FloatArray
    mode_std: float
    mix_beta: float

    def true_samples(self, c: FloatArray, n: int, seed: int) -> tuple[FloatArray, IntArray]:
        """``n`` samples of p(x|c) per condition from a fixed evaluation stream."""
        return sample_x(c, n, numpy_rng(seed, 21), mode_std=self.mode_std, mix_beta=self.mix_beta)

    def log_density(self, x: FloatArray, c: FloatArray) -> FloatArray:
        return log_density(x, c, mode_std=self.mode_std, mix_beta=self.mix_beta)

    def modes(self, c: FloatArray) -> FloatArray:
        return mode_params(c, mix_beta=self.mix_beta)[0]


def make_bank(*, n_train: int, n_heldout: int, n_eval_train: int, seed: int, mode_std: float,
              mix_beta: float) -> Bank:
    """Conditions U([0, 1]²), one positive each; held-out conditions from a separate stream."""
    c = sample_conditions(n_train, numpy_rng(seed, 20))
    y, _ = sample_x(c, 1, numpy_rng(seed, 22), mode_std=mode_std, mix_beta=mix_beta)
    eval_train = numpy_rng(seed, 23).choice(n_train, size=n_eval_train, replace=False)
    c_heldout = sample_conditions(n_heldout, numpy_rng(seed, 24))
    return Bank(c=c, y=y[:, 0], eval_train=np.sort(eval_train), c_heldout=c_heldout,
                mode_std=mode_std, mix_beta=mix_beta)
