"""Level 0/1 data and the L1 generator's capacity knobs."""

import lucid
import numpy as np

from affdrift.data import toy2d
from affdrift.data.particles import half_median_pairwise, make_gaussian_pair, make_single_bank
from affdrift.models.generator import GeneratorSpec, MLPGenerator, quantize


def test_single_bank_is_separated_and_fixed() -> None:
    a = make_single_bank(32, seed=0, box=1.0, min_sep=0.15)
    b = make_single_bank(32, seed=0, box=1.0, min_sep=0.15)
    np.testing.assert_array_equal(a.y, b.y)
    d = np.linalg.norm(a.y[:, None] - a.y[None], axis=-1) + np.eye(32) * 9
    assert d.min() >= 0.15
    np.testing.assert_allclose(a.psi[:, 0], np.linspace(0, 1, 32))


def test_gaussian_pair_moments_are_exact() -> None:
    pts = make_gaussian_pair((1.0, 0.0), (-1.0, 0.0), 0.5, 2048, seed=0)
    for p, mu in zip(pts, ((1.0, 0.0), (-1.0, 0.0))):
        np.testing.assert_allclose(p.mean(0), mu, atol=1e-12)
        np.testing.assert_allclose(np.cov(p, rowvar=False, bias=True), 0.25 * np.eye(2), atol=1e-12)


def test_sigma_rule() -> None:
    assert half_median_pairwise(np.array([[0.0, 0.0], [2.0, 0.0]])) == 1.0


def test_toy2d_mean_falls_between_modes(rng: np.random.Generator) -> None:
    c = np.array([[0.3, 0.5]])
    centres, log_pi = toy2d.mode_params(c, mix_beta=2.0)
    np.testing.assert_allclose(np.exp(log_pi), 1 / 3)
    mean = toy2d.conditional_mean(c, mix_beta=2.0)
    assert np.linalg.norm(mean) < 1e-9  # balanced modes on a circle: mean at the centre
    x, modes = toy2d.sample_x(c, 4000, rng, mode_std=0.1, mix_beta=2.0)
    assert set(np.unique(modes)) == {0, 1, 2}
    assert np.isfinite(toy2d.log_density(x, c, mode_std=0.1, mix_beta=2.0)).all()


def test_quantize_makes_cell_indistinguishable() -> None:
    c = lucid.tensor([[0.01, 0.02], [0.05, 0.06]])
    q = quantize(c, 1 / 16)
    np.testing.assert_allclose(q.numpy()[0], q.numpy()[1])
    gen = MLPGenerator(GeneratorSpec(delta=1 / 16))
    eps = lucid.randn(2, 2)
    np.testing.assert_allclose(gen(eps[:1], c[:1]).numpy(), gen(eps[:1], c[1:]).numpy(), atol=1e-6)


def test_lipschitz_generator_runs() -> None:
    gen = MLPGenerator(GeneratorSpec(lipschitz=4.0))
    out = gen(lucid.randn(5, 7, 2), lucid.rand(5, 1, 2))
    assert out.shape == (5, 7, 2)
