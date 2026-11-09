"""Level 0/1 data and the L1 generator's capacity knobs."""

import lucid
import numpy as np

from affdrift.data import toy2d
from affdrift.data.particles import half_median_pairwise, make_gaussian_pair, make_single_bank
from affdrift.models.generator import GeneratorSpec, MLPGenerator, fourier_matrix, quantize


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


def test_fourier_features_are_fixed_and_not_trained() -> None:
    """D25: B ~ N(0, s²) is a buffer drawn from the seed — same seed, same B; never a parameter."""
    spec = GeneratorSpec(fourier_scale=16.0, fourier_features=8, fourier_seed=3)
    b = fourier_matrix(spec)
    assert b.shape == (8, 2)
    np.testing.assert_array_equal(b, fourier_matrix(spec))
    assert not np.array_equal(b, fourier_matrix(GeneratorSpec(fourier_scale=16.0, fourier_features=8, fourier_seed=4)))
    np.testing.assert_allclose(b.std(), 16.0, rtol=0.6)
    gen = MLPGenerator(spec)
    n_raw = sum(int(p.numel()) for p in MLPGenerator(GeneratorSpec()).parameters())
    n_ff = sum(int(p.numel()) for p in gen.parameters())
    assert n_ff - n_raw == (2 * 8 - 2) * 64      # only the wider first layer of g; B itself is not a parameter
    assert all(p.shape != (2, 8) for p in gen.parameters())
    c = lucid.rand(5, 1, 2)
    emb = gen.cond.embed(c).numpy()
    z = 2 * np.pi * c.numpy() @ b.T
    np.testing.assert_allclose(emb, np.concatenate([np.cos(z), np.sin(z)], -1), atol=1e-4)
    assert gen(lucid.randn(5, 7, spec.eps_dim), c).shape == (5, 7, 2)


def test_fourier_scale_raises_resolution_in_c() -> None:
    """At initialisation, a larger Fourier scale makes f vary faster in c (the knob binds, D25)."""
    eps = lucid.zeros(1, 1, 2)
    c0 = lucid.tensor([[[0.5, 0.5]]])
    c1 = lucid.tensor([[[0.501, 0.5]]])

    def change(s: float | None) -> float:
        diffs = []
        for seed in range(8):
            lucid.manual_seed(seed)
            g = MLPGenerator(GeneratorSpec(fourier_scale=s, fourier_seed=seed))
            diffs.append(float(np.linalg.norm((g(eps, c1) - g(eps, c0)).numpy())))
        return float(np.mean(diffs))

    assert change(64.0) > 5 * change(None)
