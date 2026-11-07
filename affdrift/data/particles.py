r"""Level 0: non-parametric particles (plan §4).

Each condition owns ``M`` particles that move by :math:`x \leftarrow x + \eta V(x\mid c)`,
so the "generator" has unlimited capacity and the run checks the field's fixed
points alone.  Two problems:

* **single bank** (E0.2–E0.4, E0.6) — ``K`` conditions on a line, one
  positive :math:`y_k \in \mathbb R^2` each; positives are borrowed through
  caption weights :math:`\tilde w_j(c_k)`.
* **Gaussian pair** (E0.5) — two conditions with Gaussian positives that are
  each other's counterfactual negative.
"""

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import lucid
import numpy as np
from lucid import Tensor
from scipy.spatial.distance import pdist

from affdrift._typing import FloatArray
from affdrift.core.drift import AffinityDrift, DriftLoss, DriftTerms, TargetSet
from affdrift.runtime import numpy_rng, to_numpy

TERMS = ("v", "v_plus", "v_minus", "v_neg")

type Field = Callable[[Tensor], DriftTerms]


# ── data ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SingleBank:
    """``K`` conditions, :math:`\\psi_k` evenly spaced on [0, 1], one :math:`y_k` each."""

    psi: FloatArray  # (K, 1)
    y: FloatArray    # (K, 2)


def make_single_bank(k: int, *, seed: int, box: float, min_sep: float, max_tries: int = 100_000) -> SingleBank:
    """:math:`y_k \\sim U([-b, b]^2)` with rejection so that every pair is ``min_sep`` apart."""
    rng = numpy_rng(seed, 0)
    points: list[FloatArray] = []
    for _ in range(max_tries):
        cand = rng.uniform(-box, box, size=2)
        if all(np.linalg.norm(cand - p) >= min_sep for p in points):
            points.append(cand)
            if len(points) == k:
                return SingleBank(psi=np.linspace(0.0, 1.0, k)[:, None], y=np.stack(points))
    raise RuntimeError(f"could not place {k} points {min_sep} apart in [-{box}, {box}]²")


def make_gaussian_pair(
    mu1: tuple[float, ...], mu2: tuple[float, ...], s: float, n: int, *, seed: int
) -> FloatArray:
    """``(2, n, D)`` positives whose sample mean is exactly μ and sample covariance exactly s² I."""
    rng = numpy_rng(seed, 1)
    out = []
    for mu in (mu1, mu2):
        z = rng.normal(size=(n, len(mu)))
        z -= z.mean(axis=0)
        chol = np.linalg.cholesky(np.cov(z, rowvar=False, bias=True))
        z = np.linalg.solve(chol, z.T).T
        out.append(np.asarray(mu) + s * z)
    return np.stack(out)


def init_particles(k: int, m: int, *, seed: int, box: float, dim: int = 2) -> FloatArray:
    return numpy_rng(seed, 2).uniform(-box, box, size=(k, m, dim))


def half_median_pairwise(points: FloatArray) -> float:
    """σ rule of B0 (D5): half the median pairwise distance of the positive bank."""
    return 0.5 * float(np.median(pdist(np.asarray(points).reshape(-1, points.shape[-1]))))


# ── dynamics ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True, kw_only=True)
class ParticleProblem:
    """The static half of a particle simulation.

    Attributes
    ----------
    pos
        Positives and their caption log-weights, ``log_w`` of shape ``(K, 1, P)``.
    rep_log_w
        ``None`` for repulsion from the condition's own particles
        (:math:`h_q = 0`), else ``(K, M, K·M)`` log-weights over *all*
        particles for :math:`h_q > 0`.
    exclude_self
        Leave-one-out repulsion (B0).  Level 0 sets it ``False``: for a fixed
        particle system the empirical KDE including the particle is the
        faithful q̂ (docs/deviations.md D6).
    partner
        Counterfactual partner of each condition (E0.5), or ``None``.
    """

    pos: TargetSet
    rep_log_w: Tensor | None = None
    partner: tuple[int, ...] | None = None
    exclude_self: bool = True

    def gen_targets(self, x: Tensor) -> TargetSet:
        if self.rep_log_w is None:
            return TargetSet.own_samples(x, exclude_self=self.exclude_self)
        k, m, d = (int(s) for s in x.shape)
        return TargetSet(x.reshape(1, k * m, d), self.rep_log_w)

    def neg_targets(self, x: Tensor) -> TargetSet | None:
        if self.partner is None:
            return None
        return TargetSet.uniform(lucid.stack([x[j] for j in self.partner]))


@dataclass(slots=True)
class Trajectory:
    """What :meth:`ParticleSystem.run` returns."""

    final: FloatArray
    steps: int
    snapshots: dict[int, FloatArray]
    checks: list[dict[str, float]]
    means: list[FloatArray]
    converged_at: int | None = None
    stable: bool = False
    diverged: bool = False
    compiled: bool = False
    wall_time: float = 0.0
    notes: list[str] = field(default_factory=list)


def snapshot_steps(max_steps: int, ratio: float = 1.5) -> set[int]:
    """0 and a geometric grid — enough for any "start / middle / end" view."""
    steps, t = {0}, 1.0
    while t <= max_steps:
        steps.add(int(round(t)))
        t *= ratio
    return steps


def affd_field(problem: ParticleProblem, drift: AffinityDrift) -> Field:
    """AffD's field for a particle problem, term by term."""
    return lambda x: drift.field(x, problem.pos, problem.gen_targets(x), problem.neg_targets(x))


def release_field(positives: Tensor, drift: DriftLoss) -> Field:
    """The original Drifting field (baseline arm): each condition's own positives, own particles as negatives."""
    return lambda x: DriftTerms(v=drift.drift(x, positives))


def gaussian_closure_field(
    mu: Tensor, s: float, sigma: float, lam: float, partner: tuple[int, ...] | None
) -> Field:
    r"""E0.5b diagnostic: Lemma 1's field with every KDE replaced by its Gaussian (docs/deviations.md D13).

    .. math::

        V = \sigma^2\big[-\tfrac{x-\mu_c}{\tilde s^2} + \tfrac{x-m_c}{v_c+\sigma^2}
            + \lambda\,\tfrac{x-m_{c'}}{v_{c'}+\sigma^2}\big]

    with :math:`m, v` the current particle mean and per-dimension variance.  It
    keeps Prop E's algebra and removes the finite-sample tails of p̂ and q̂.
    """
    s2_tilde = s**2 + sigma**2

    def field(x: Tensor) -> DriftTerms:
        m = x.mean(dim=1, keepdim=True)
        v = ((x - m) ** 2).mean(dim=(1, 2), keepdim=True)
        v_plus = (mu.unsqueeze(1) - x) * (sigma**2 / s2_tilde)
        v_minus = (m - x) * sigma**2 / (v + sigma**2)
        if partner is None or lam == 0:
            return DriftTerms(v=v_plus - v_minus, v_plus=v_plus, v_minus=v_minus)
        mp = lucid.stack([m[j] for j in partner])
        vp = lucid.stack([v[j] for j in partner])
        v_neg = (mp - x) * sigma**2 / (vp + sigma**2)
        return DriftTerms(v=v_plus - v_minus - lam * v_neg, v_plus=v_plus, v_minus=v_minus, v_neg=v_neg)

    return field


class ParticleSystem:
    """Integrates :math:`x \\leftarrow x + \\eta V` for every condition at once.

    ``field`` maps the ``(K, M, D)`` particles to their :class:`DriftTerms`;
    AffD and the baseline share this integrator, so they differ only in the
    field.  The step is compiled with :func:`lucid.compile` when asked and
    when the compiled step matches the eager one on the initial state;
    otherwise it runs eagerly and says so in :attr:`Trajectory.notes`.
    """

    def __init__(self, field: Field, *, eta: float) -> None:
        self.field = field
        self.eta = eta

    def step(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """``(x + ηV, [rms v, v⁺, v⁻, v_neg, max |x|])``."""
        terms = self.field(x)
        rms = terms.rms()
        zero = lucid.zeros((), dtype=x.dtype, device=x.device.type)
        stats = lucid.stack([rms.get(name, zero) for name in TERMS] + [lucid.abs(x).max()])
        return x + self.eta * terms.v, stats

    def _stepper(self, x0: Tensor, want_compile: bool, notes: list[str]) -> tuple[Callable[[Tensor], tuple[Tensor, Tensor]], bool]:
        if not want_compile:
            return self.step, False
        try:
            compiled = lucid.compile(self.step)
            ex, es = self.step(x0)
            cx, cs = compiled(x0)
            err = float(lucid.abs(ex - cx).max().item())
            scale = max(1.0, float(lucid.abs(ex).max().item()))
        except Exception as exc:  # noqa: BLE001 — fall back to eager, recorded
            notes.append(f"lucid.compile failed ({type(exc).__name__}: {exc}); eager")
            return self.step, False
        if err > 1e-5 * scale:
            notes.append(f"compiled step differs from eager by {err:.2e}; eager")
            return self.step, False
        notes.append(f"compiled step matches eager (max diff {err:.1e})")
        return compiled, True

    def run(
        self,
        x0: Tensor,
        *,
        max_steps: int,
        tol: float,
        stable_window: int,
        check_every: int,
        diverge_bound: float,
        compile: bool = True,
        on_check: Callable[[int, Mapping[str, float]], None] | None = None,
    ) -> Trajectory:
        notes: list[str] = []
        stepper, compiled = self._stepper(x0, compile, notes)
        snaps_at = snapshot_steps(max_steps)
        snapshots: dict[int, FloatArray] = {}
        checks: list[dict[str, float]] = []
        means: list[FloatArray] = []
        below_since: int | None = None
        stable = diverged = False
        x, t, start = x0, 0, time.perf_counter()

        while True:
            if t in snaps_at:
                snapshots[t] = to_numpy(x)
            if t >= max_steps:
                break
            x_next, stats = stepper(x)
            lucid.eval(x_next, stats)  # Metal is lazy (MLX): materialise every step, or the graph grows
            if t % check_every == 0:
                values = to_numpy(stats)
                row = {f"rms_{name}": float(v) for name, v in zip(TERMS, values[:4], strict=True)}
                row |= {"max_abs_x": float(values[4])}
                checks.append({"step": t, **row})
                means.append(to_numpy(x.mean(dim=1)))
                if on_check is not None:
                    on_check(t, row)
                if not np.all(np.isfinite(values)) or values[4] > diverge_bound:
                    diverged = True
                    break
                if values[0] < tol:
                    below_since = t if below_since is None else below_since
                    if t - below_since >= stable_window:
                        stable = True
                        break
                else:
                    below_since = None
            x, t = x_next, t + 1

        snapshots[t] = to_numpy(x)
        return Trajectory(
            final=snapshots[t], steps=t, snapshots=snapshots, checks=checks, means=means,
            converged_at=below_since, stable=stable, diverged=diverged, compiled=compiled,
            wall_time=time.perf_counter() - start, notes=notes,
        )
