"""Particle dynamics: compiled step equals eager, and the simplest fixed points are reached."""

import numpy as np

from affdrift.core.drift import AffinityDrift, DriftSpec, TargetSet
from affdrift.data.particles import ParticleProblem, ParticleSystem, affd_field, make_gaussian_pair
from affdrift.runtime import to_numpy, to_tensor


def _system(device: str, y: np.ndarray, m: int = 64, *, exclude_self: bool = False) -> tuple[ParticleSystem, object]:
    pos = TargetSet.uniform(to_tensor(y, device))
    problem = ParticleProblem(pos=pos, exclude_self=exclude_self)
    system = ParticleSystem(affd_field(problem, AffinityDrift(DriftSpec(sigma=0.4, drift_norm=False))), eta=0.5)
    x0 = to_tensor(np.random.default_rng(0).uniform(-1.5, 1.5, size=(len(y), m, 2)), device)
    return system, x0


def test_compiled_step_equals_eager(device: str) -> None:
    import lucid

    y = make_gaussian_pair((1.0, 0.0), (-1.0, 0.0), 0.3, 64, seed=0)
    system, x0 = _system(device, y)
    ex, es = system.step(x0)
    cx, cs = lucid.compile(system.step)(x0)
    assert np.abs(to_numpy(ex) - to_numpy(cx)).max() < 1e-5
    assert np.abs(to_numpy(es) - to_numpy(cs)).max() < 1e-5


def test_lambda0_pair_matches_positive_moments(device: str) -> None:
    """With h_q = 0, no negative and the full empirical KDE (L0, D6), q̂ = p̂: the positives' moments."""
    y = make_gaussian_pair((1.0, 0.0), (-1.0, 0.0), 0.3, 256, seed=0)
    system, x0 = _system(device, y, m=256)
    traj = system.run(x0, max_steps=4000, tol=1e-6, stable_window=100, check_every=50, diverge_bound=1e3,
                      compile=False)
    for c, mu in enumerate(((1.0, 0.0), (-1.0, 0.0))):
        np.testing.assert_allclose(traj.final[c].mean(0), mu, atol=0.02)
        assert abs(traj.final[c].var(0).mean() - 0.09) / 0.09 < 0.1


def test_leave_one_out_biases_fixed_particles(device: str) -> None:
    """The finding behind D6: self-excluded repulsion over-disperses a fixed particle system."""
    y = make_gaussian_pair((1.0, 0.0), (-1.0, 0.0), 0.3, 256, seed=0)
    system, x0 = _system(device, y, m=64, exclude_self=True)
    traj = system.run(x0, max_steps=4000, tol=1e-7, stable_window=100, check_every=50, diverge_bound=1e3,
                      compile=False)
    assert traj.final[0].var(0).mean() > 0.09 * 1.3
