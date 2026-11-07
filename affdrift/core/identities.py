r"""Closed-form identities the field must satisfy, as executable references.

* **Lemma 0** (G2, E1.1): with one positive and no repulsion, the drifting
  loss has the gradient of :math:`\lVert f - y_i\rVert^2`.
* **Lemma 1** (E0.1): for a Gaussian :math:`k_x`,
  :math:`V = \sigma^2 \nabla_x[\log \hat p(x\mid c) - \log \hat q(x\mid c)]`,
  where :math:`\hat q` is the leave-one-out KDE of the generated samples.
* **Prop E** (E0.5): the equilibrium of two conditions that are each other's
  counterfactual negative with weight :math:`\lambda < 1`.
"""

from dataclasses import dataclass

import lucid
import lucid.autograd
from lucid import Tensor

from affdrift.core.drift import TargetSet
from affdrift.core.kernels import sqdist


def score_ratio_field(x: Tensor, pos: TargetSet, gen: TargetSet, sigma: float) -> Tensor:
    r""":math:`\sigma^2\nabla_x[\log\hat p - \log\hat q]` by autograd (Lemma 1's right-hand side).

    ``gen.points`` are held fixed (detached), so only the query ``x`` moves,
    as in the field.
    """
    query = x.detach().requires_grad_(True)
    inv = -0.5 / sigma**2
    log_p = lucid.logsumexp(pos.log_w + sqdist(query, pos.points.detach()) * inv, dim=-1)
    log_q = lucid.logsumexp(gen.log_w + sqdist(query, gen.points.detach()) * inv, dim=-1)
    (grad,) = lucid.autograd.grad((log_p - log_q).sum(), query)
    return grad * sigma**2


@dataclass(frozen=True, slots=True)
class GaussianEquilibrium:
    """Prop E's prediction for one condition of a counterfactual pair."""

    mean: tuple[float, ...]
    #: Per-dimension variance of the *smoothed* density q̂ = q ∗ N(0, σ²).
    smoothed_var: float
    #: Per-dimension variance of the particles themselves, smoothed_var − σ².
    particle_var: float


def prop_e_equilibrium(
    mu1: tuple[float, ...], mu2: tuple[float, ...], s: float, sigma: float, lam: float
) -> GaussianEquilibrium | None:
    r"""Equilibrium of :math:`\log\hat q_c = \log\hat p_c - \lambda\log\hat q_{c'}` for condition ``c``.

    :math:`\hat q_c = \mathcal N((\mu_1 - \lambda\mu_2)/(1-\lambda), (1+\lambda)\tilde s^2)` with
    :math:`\tilde s^2 = s^2 + \sigma^2`.  ``None`` for :math:`\lambda \ge 1`, where
    the plan predicts divergence.
    """
    if lam >= 1:
        return None
    s2_tilde = s**2 + sigma**2
    smoothed = (1 + lam) * s2_tilde
    mean = tuple((a - lam * b) / (1 - lam) for a, b in zip(mu1, mu2, strict=True))
    return GaussianEquilibrium(mean=mean, smoothed_var=smoothed, particle_var=smoothed - sigma**2)
