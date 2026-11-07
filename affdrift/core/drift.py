r"""The drifting field: the release loss, and Affinity Drifting as its subclass.

:class:`DriftLoss` is the drifting loss of Deng et al. (arXiv 2602.04770)
exactly as ``lucid-dl==3.16.4`` ships it
(``lucid.models.generative.drifting._model._drift_loss``).  Lucid implements
it as module functions; this class calls those same functions, one per
overridable step, so that its output equals Lucid's bit for bit
(``tests/core/test_drift.py``).

:class:`AffinityDrift` inherits it and overrides only what AffD changes:

* :meth:`~AffinityDrift.distances` — squared distances for a Gaussian
  :math:`k_x` with one frozen :math:`\sigma` (CLAUDE.md §2.3), instead of a
  Laplace kernel at several temperatures;
* :meth:`~AffinityDrift.scale` — a frozen feature scale instead of one
  re-estimated every batch (docs/deviations.md D3);
* :meth:`~AffinityDrift.goal` — the field of plan §4 / CLAUDE.md §4:
  caption-weighted mean-shift attraction, repulsion over generated samples
  with caption bandwidth :math:`h_q`, an optional counterfactual negative
  :math:`\lambda`, and per-term drift normalisation.

The training loss (:meth:`DriftLoss.forward`, an MSE onto the stop-gradient
drifted target) is inherited unchanged.  The single-positive arm and the
controls K1–K5 are options of this one class (CLAUDE.md §2.2):

=========  =========================================================
single     caption weights one-hot (``h_p = 0``)
K1         ``kernel = XKernel.NONE``
K2         weights from :func:`~affdrift.core.kernels.shuffle_neighbors`
K3         ``pos_noise_std > 0``
K4         oracle positives in the :class:`TargetSet`
K5         uniform caption weights (``h_p = inf``)
=========  =========================================================
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Self, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid import Tensor
from lucid._dtype import _float32_or_wider
from lucid.models.generative.drifting import _model as release

from affdrift.core.kernels import XKernel, log_x_kernel, sqdist

__all__ = ["AffinityDrift", "DriftLoss", "DriftSpec", "DriftTerms", "RawDriftLoss", "Softmax", "TargetSet"]

type Info = dict[str, Tensor]


# ─────────────────────────────────────────────────────────────────────────────
# The release loss
# ─────────────────────────────────────────────────────────────────────────────


class DriftLoss(nn.Module):
    """The release drifting loss, one overridable method per step.

    Parameters
    ----------
    temperatures
        The Laplace-kernel temperatures :math:`\\tau`; one normalised drift
        each, summed.  The release default is ``(0.02, 0.05, 0.2)``.
    """

    def __init__(self, temperatures: Sequence[float] = (0.02, 0.05, 0.2)) -> None:
        super().__init__()
        self.temperatures = tuple(float(t) for t in temperatures)

    # ── steps ────────────────────────────────────────────────────────────

    def targets(
        self, old: Tensor, pos: Tensor, neg: Tensor | None, weight_neg: Tensor | None
    ) -> tuple[Tensor, Tensor, int]:
        """Targets ordered ``[generated, unconditional, positive]`` and their weights."""
        return release._drift_targets(old, pos, neg, weight_neg)

    def self_mask(self, old: Tensor, targets: Tensor) -> Tensor:
        """One where a generated sample meets itself among the targets."""
        return release._self_mask(old, targets)

    def distances(self, old: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        """``(B, C, N)`` L2 distances, self pairs exactly zero."""
        return release._pairwise_distance(old, targets, mask)

    def scale(self, dist: Tensor, target_w: Tensor) -> Tensor:
        """The weighted mean distance, re-estimated on every batch (A.6)."""
        return (dist * target_w.unsqueeze(1)).mean() / target_w.mean()

    def force(
        self,
        dist_normed: Tensor,
        mask: Tensor,
        target_w: Tensor,
        targets_scaled: Tensor,
        old_scaled: Tensor,
        c_n: int,
    ) -> tuple[Tensor, Info]:
        """Alg. 2's field, normalised per temperature and summed."""
        return release._drift_force(
            dist_normed, mask, target_w, targets_scaled, old_scaled, c_n, self.temperatures
        )

    def goal(
        self, old: Tensor, pos: Tensor, neg: Tensor | None = None, weight_neg: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Info]:
        """The drifted target in normalised coordinates, the input scale, and diagnostics."""
        s = int(old.shape[-1])
        targets, target_w, c_n = self.targets(old, pos, neg, weight_neg)
        mask = self.self_mask(old, targets)
        dist = self.distances(old, targets, mask)
        scale = self.scale(dist, target_w)
        scale_inputs = lucid.clamp(scale / math.sqrt(s), min=1e-3)
        dist_normed = dist / lucid.clamp(scale, min=1e-3)
        force, energies = self.force(
            dist_normed, mask, target_w, targets / scale_inputs, old / scale_inputs, c_n
        )
        return old / scale_inputs + force, scale_inputs, {"scale": scale, **energies}

    def drift(self, old: Tensor, *args: object, **kwargs: object) -> Tensor:
        """The field V in input coordinates, ``(goal − old/s)·s`` — what a particle moves along."""
        goal, scale_inputs, _ = self.goal(old, *args, **kwargs)
        return (goal - old / scale_inputs) * scale_inputs

    # ── the loss ─────────────────────────────────────────────────────────

    @override
    def forward(
        self,
        gen: Tensor,
        pos: Tensor,
        neg: Tensor | None = None,
        weight_neg: Tensor | None = None,
        **goal_kwargs: object,
    ) -> tuple[Tensor, Info]:
        """``(B,)`` per-group loss :math:`\\lVert f - \\mathrm{sg}(f + V)\\rVert^2` and diagnostics.

        Everything up to the target runs without gradient; the gradient
        reaches the generator only through ``gen`` itself.
        """
        gen = gen.to(_float32_or_wider(gen.dtype))
        with lucid.no_grad():
            goal, scale_inputs, info = self.goal(gen.detach(), pos, neg, weight_neg, **goal_kwargs)
        diff = gen / scale_inputs - goal
        return (diff * diff).mean(dim=(1, 2)), info


class RawDriftLoss(DriftLoss):
    """The release field without its per-temperature drift normalisation.

    The baseline arm ``drifting`` of Level 0, where every arm integrates the
    raw field (docs/deviations.md D4): normalisation rescales V but leaves its
    zeros — the fixed points L0 measures — where they are.  The release
    function is still what computes the field: it is called once per
    temperature and its normalisation undone with the energy it reports, so
    no line of the release is re-implemented.
    """

    @override
    def force(
        self,
        dist_normed: Tensor,
        mask: Tensor,
        target_w: Tensor,
        targets_scaled: Tensor,
        old_scaled: Tensor,
        c_n: int,
    ) -> tuple[Tensor, Info]:
        total = lucid.zeros_like(old_scaled)
        energies: Info = {}
        for tau in self.temperatures:
            force, energy = release._drift_force(
                dist_normed, mask, target_w, targets_scaled, old_scaled, c_n, (tau,)
            )
            raw_energy = energy[f"loss_{tau}"]
            energies[f"loss_{tau}"] = raw_energy
            total = total + force * lucid.sqrt(lucid.clamp(raw_energy, min=1e-8))
        return total, energies


# ─────────────────────────────────────────────────────────────────────────────
# Affinity Drifting
# ─────────────────────────────────────────────────────────────────────────────


class Softmax(StrEnum):
    """How attraction and repulsion weights are normalised (E0.6)."""

    #: Each term is its own mean-shift (B0): V = V⁺ − V⁻.
    SEPARATE = "separate"
    #: One softmax over [generated, positive] with Alg. 2's cross weighting,
    #: V = Z⁺Z⁻ (V⁺ − V⁻) for the row's group masses Z⁺, Z⁻.
    JOINT = "joint"


@dataclass(frozen=True, slots=True, kw_only=True)
class DriftSpec:
    """Every knob of the AffD field.  One instance per arm."""

    sigma: float
    kernel: XKernel = XKernel.GAUSSIAN
    softmax: Softmax = Softmax.SEPARATE
    repulsion: bool = True
    neg_lambda: float = 0.0
    drift_norm: bool = True
    feature_scale: float = 1.0
    pos_noise_std: float = 0.0

    def __post_init__(self) -> None:
        if not self.sigma > 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if not self.feature_scale > 0:
            raise ValueError(f"feature_scale must be positive, got {self.feature_scale}")
        if self.neg_lambda < 0 or self.pos_noise_std < 0:
            raise ValueError("neg_lambda and pos_noise_std must be non-negative")


@dataclass(frozen=True, slots=True)
class TargetSet:
    """Points a sample is compared with, and their log-weights.

    Attributes
    ----------
    points
        ``(B, N, S)``, or ``(1, N, S)`` shared by every group.
    log_w
        Broadcastable to ``(B, C, N)``: per group and query, the log-weight
        of each point.  ``-inf`` removes a point (e.g. a sample itself).
        Rows should be normalised over the finite entries, so each set is a
        distribution — required for :attr:`Softmax.JOINT`.
    """

    points: Tensor
    log_w: Tensor

    @classmethod
    def uniform(cls, points: Tensor) -> Self:
        """Equal weight on every point of each group."""
        n = int(points.shape[-2])
        return cls(points, lucid.full((1, 1, n), -math.log(n), dtype=points.dtype, device=points.device.type))

    @classmethod
    def own_samples(cls, x: Tensor, *, exclude_self: bool = True) -> Self:
        """A group's own samples (``h_q = 0``).

        ``exclude_self=True`` (B0, CLAUDE.md §4, and the release's ``mask_val``)
        drops each sample from its own row: a leave-one-out KDE, unbiased for
        i.i.d. generator samples.  ``False`` keeps it: the plain empirical KDE.
        """
        c = int(x.shape[-2])
        if not exclude_self:
            return cls.uniform(x)
        if c < 2:
            raise ValueError("repulsion over own samples needs at least two samples per group")
        eye = lucid.eye(c, dtype=x.dtype, device=x.device.type)
        log_w = (1.0 - eye) * -math.log(c - 1) + eye * -1e30
        return cls(x, log_w.unsqueeze(0).masked_fill(eye.unsqueeze(0) > 0, float("-inf")))

    def scaled(self, factor: float) -> Self:
        return replace(self, points=self.points * factor)


@dataclass(frozen=True, slots=True)
class DriftTerms:
    """The field and its terms, all ``(B, C, S)``; absent terms are ``None``."""

    v: Tensor
    v_plus: Tensor | None = None
    v_minus: Tensor | None = None
    v_neg: Tensor | None = None

    def rms(self) -> Mapping[str, Tensor]:
        """Root-mean-square norm of each present term over groups and samples."""
        return {
            name: lucid.sqrt((term * term).sum(dim=-1).mean())
            for name in ("v", "v_plus", "v_minus", "v_neg")
            if (term := getattr(self, name)) is not None
        }


def _as_set(value: Tensor | TargetSet | None) -> TargetSet | None:
    match value:
        case None | TargetSet():
            return value
        case _:
            return TargetSet.uniform(value)


def _unit_energy(v: Tensor) -> Tensor:
    """Scale so the batch has E‖V‖²/S = 1 (CLAUDE.md §4, the release's per-τ normalisation)."""
    return v / lucid.sqrt(lucid.clamp((v * v).mean(), min=1e-16))


class AffinityDrift(DriftLoss):
    r"""AffD's field and loss.  See the module docstring for what is overridden.

    The field at samples :math:`x` of a group with condition :math:`c` is

    .. math::

        V^+ = \sum_i \mathrm{softmax}_i\big(\log w_i(c) + \log k_x(x, y_i)\big)(y_i - x),

    :math:`V^-` the same over generated samples with caption bandwidth
    :math:`h_q`, :math:`V_\mathrm{neg}` the same over counterfactual
    negatives, and :math:`V = V^+ - V^- - \lambda V_\mathrm{neg}`.

    Parameters
    ----------
    spec
        The arm's knobs.
    generator
        Lucid RNG for control K3's positive noise; required only then.
    """

    def __init__(self, spec: DriftSpec, *, generator: lucid.Generator | None = None) -> None:
        super().__init__(temperatures=(spec.sigma,))
        if spec.pos_noise_std > 0 and generator is None:
            raise ValueError("K3 (pos_noise_std > 0) needs an explicit generator")
        self.spec = spec
        self.generator = generator

    # ── overridden steps ─────────────────────────────────────────────────

    @override
    def distances(self, old: Tensor, targets: Tensor, mask: Tensor | None = None) -> Tensor:
        """Squared distances; exclusions live in :attr:`TargetSet.log_w` instead of ``mask``."""
        return sqdist(old, targets)

    @override
    def scale(self, dist: Tensor | None = None, target_w: Tensor | None = None) -> Tensor:
        """The frozen feature scale from ``sigma.json`` — never re-estimated per batch."""
        return lucid.full((), self.spec.feature_scale)

    @override
    def goal(
        self,
        old: Tensor,
        pos: Tensor | TargetSet,
        neg: Tensor | TargetSet | None = None,
        weight_neg: Tensor | None = None,
        *,
        gen_targets: TargetSet | None = None,
    ) -> tuple[Tensor, Tensor, Info]:
        """``old`` drifted by :meth:`field`, in frozen-scale coordinates.

        ``gen_targets`` defaults to the group's own samples (``h_q = 0``).
        """
        if weight_neg is not None:
            raise TypeError("AffD weighs negatives by spec.neg_lambda and TargetSet.log_w, not weight_neg")
        scale = self.scale()
        inv = 1.0 / self.spec.feature_scale
        x = old * inv
        pos_set, neg_set = _as_set(pos), _as_set(neg)
        terms = self.field(
            x,
            pos_set.scaled(inv),
            None if gen_targets is None else gen_targets.scaled(inv),
            None if neg_set is None else neg_set.scaled(inv),
        )
        return x + terms.v, scale, {f"rms_{k}": v for k, v in terms.rms().items()}

    # ── the field ────────────────────────────────────────────────────────

    def logits(self, x: Tensor, targets: TargetSet) -> Tensor:
        r"""``(B, C, N)`` :math:`\log w + \log k_x(x, \cdot)`."""
        d2 = self.distances(x, targets.points)
        return targets.log_w + log_x_kernel(d2, kind=self.spec.kernel, sigma=self.spec.sigma)

    def mean_shift(self, x: Tensor, targets: TargetSet) -> Tensor:
        r""":math:`\sum_j a_j (y_j - x)` with :math:`a = \mathrm{softmax}(\text{logits})`."""
        a = F.softmax(self.logits(x, targets), dim=-1)
        return lucid.matmul(a, targets.points) - x

    def perturb(self, pos: TargetSet) -> TargetSet:
        """Control K3: Gaussian noise on the positives, fresh every call."""
        if self.spec.pos_noise_std == 0:
            return pos
        noise = lucid.randn(
            *pos.points.shape, dtype=pos.points.dtype, device=pos.points.device.type, generator=self.generator
        )
        return replace(pos, points=pos.points + noise * self.spec.pos_noise_std)

    def field(
        self,
        x: Tensor,
        pos: TargetSet,
        gen: TargetSet | None = None,
        neg: TargetSet | None = None,
    ) -> DriftTerms:
        """The drifting field at ``x`` (``(B, C, S)``), term by term."""
        spec = self.spec
        if spec.repulsion and gen is None:
            gen = TargetSet.own_samples(x)
        if not spec.repulsion:
            gen = None
        pos = self.perturb(pos)

        match spec.softmax:
            case Softmax.SEPARATE:
                v_plus = self.mean_shift(x, pos)
                v_minus = None if gen is None else self.mean_shift(x, gen)
                v_neg = None if neg is None or spec.neg_lambda == 0 else self.mean_shift(x, neg)
                if spec.drift_norm:
                    v_plus, v_minus, v_neg = (
                        None if t is None else _unit_energy(t) for t in (v_plus, v_minus, v_neg)
                    )
                v = v_plus
                if v_minus is not None:
                    v = v - v_minus
                if v_neg is not None:
                    v = v - spec.neg_lambda * v_neg
                return DriftTerms(v=v, v_plus=v_plus, v_minus=v_minus, v_neg=v_neg)

            case Softmax.JOINT:
                if neg is not None and spec.neg_lambda > 0:
                    raise NotImplementedError("joint softmax with counterfactual negatives is not defined in the plan")
                if gen is None:
                    v_plus = self.mean_shift(x, pos)
                    return DriftTerms(v=_unit_energy(v_plus) if spec.drift_norm else v_plus, v_plus=v_plus)
                n_gen = int(self.logits(x, gen).shape[-1])
                a = F.softmax(lucid.cat([self.logits(x, gen), self.logits(x, pos)], dim=-1), dim=-1)
                a_gen, a_pos = a[..., :n_gen], a[..., n_gen:]
                z_gen = a_gen.sum(dim=-1, keepdim=True)
                z_pos = a_pos.sum(dim=-1, keepdim=True)
                v_plus = z_gen * (lucid.matmul(a_pos, pos.points) - z_pos * x)
                v_minus = z_pos * (lucid.matmul(a_gen, gen.points) - z_gen * x)
                v = v_plus - v_minus
                return DriftTerms(v=_unit_energy(v) if spec.drift_norm else v, v_plus=v_plus, v_minus=v_minus)
