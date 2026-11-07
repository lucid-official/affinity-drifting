"""Caption kernel limits, K2 shuffling and the N_eff bandwidth rule."""

import numpy as np
import pytest

from affdrift.core.bandwidth import n_eff, solve_hp
from affdrift.core.kernels import caption_log_weights, shuffle_neighbors


def test_weights_are_normalised(rng: np.random.Generator) -> None:
    psi = rng.uniform(size=(20, 2))
    w = np.exp(caption_log_weights(psi, psi, rng.uniform(0.05, 0.5, size=20)))
    np.testing.assert_allclose(w.sum(axis=1), 1.0)


def test_hp_infinite_is_uniform_k5(rng: np.random.Generator) -> None:
    psi = rng.uniform(size=(7, 1))
    np.testing.assert_allclose(np.exp(caption_log_weights(psi, psi, np.inf)), 1 / 7)


def test_k2_shuffle_keeps_n_eff(rng: np.random.Generator) -> None:
    psi = rng.uniform(size=(30, 1))
    log_w = caption_log_weights(psi, psi, 0.1)
    shuffled = shuffle_neighbors(log_w, rng)
    np.testing.assert_allclose(n_eff(shuffled), n_eff(log_w))
    assert not np.allclose(shuffled, log_w)


@pytest.mark.parametrize("n_star", [2.0, 8.0, 16.0, 31.0])
def test_solve_hp_reaches_n_star_minimally(n_star: float) -> None:
    psi = np.linspace(0, 1, 32)[:, None]
    h = solve_hp(psi, psi, n_star)
    assert np.all(n_eff(caption_log_weights(psi, psi, h)) >= n_star - 1e-6)
    assert np.all(n_eff(caption_log_weights(psi, psi, h * (1 - 1e-6))) < n_star)


def test_n_eff_is_monotone_in_h() -> None:
    psi = np.random.default_rng(0).uniform(size=(50, 1))
    hs = np.geomspace(1e-3, 10, 200)
    ne = np.stack([n_eff(caption_log_weights(psi[:5], psi, h)) for h in hs])
    assert np.all(np.diff(ne, axis=0) >= -1e-9)


def test_solve_hp_rejects_unreachable_n_star() -> None:
    with pytest.raises(ValueError):
        solve_hp(np.zeros((1, 1)), np.zeros((3, 1)), 4)
