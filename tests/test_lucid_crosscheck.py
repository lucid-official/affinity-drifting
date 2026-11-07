"""Cross-checks of the Lucid operations L1 relies on, against independent NumPy (float64) references.

Lucid is our framework and can have defects like any library; every operation a verdict depends on is checked
here before any L1 run (user instruction, 2026-11-08).
"""

import lucid
import lucid.nn as nn
import numpy as np
import pytest
from lucid.optim import Adam
from lucid.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from scipy.special import softmax

from affdrift.core.drift import AffinityDrift, DriftSpec, TargetSet
from affdrift.models.generator import GeneratorSpec, MLPGenerator, quantize
from affdrift.runtime import to_numpy, to_tensor


def _np_mean_shift(x: np.ndarray, pts: np.ndarray, log_w: np.ndarray, sigma: float) -> np.ndarray:
    d2 = ((x[:, :, None, :] - pts[:, None, :, :]) ** 2).sum(-1)
    a = softmax(log_w - d2 / (2 * sigma**2), axis=-1)
    return np.einsum("bcn,bnd->bcd", a, pts) - x


def test_affd_field_matches_numpy_on_device(rng: np.random.Generator, device: str) -> None:
    b, c, n, s, sigma = 4, 16, 200, 2, 0.3
    x = rng.normal(size=(b, c, s))
    y = rng.normal(size=(b, n, s))
    lw = rng.normal(size=(b, n))
    lw -= np.log(np.exp(lw).sum(-1, keepdims=True))
    v_plus = _np_mean_shift(x, y, lw[:, None, :], sigma)
    own = np.where(np.eye(c, dtype=bool), -np.inf, -np.log(c - 1))[None]
    v_minus = _np_mean_shift(x, x, np.broadcast_to(own, (b, c, c)), sigma)
    ref = v_plus - v_minus
    drift = AffinityDrift(DriftSpec(sigma=sigma, drift_norm=False))
    got = drift.field(to_tensor(x, device), TargetSet(to_tensor(y, device), to_tensor(lw[:, None, :], device))).v
    assert np.abs(to_numpy(got) - ref).max() < 2e-5


def test_adam_step_matches_numpy() -> None:
    w0 = np.array([[0.5, -1.0], [2.0, 0.1]])
    g = np.array([[0.3, -0.2], [0.05, 1.0]])
    p = nn.Parameter(lucid.tensor(w0.astype(np.float32)))
    opt = Adam([p], lr=1e-2)
    for _ in range(3):
        p.grad = lucid.tensor(g.astype(np.float32))
        opt.step()
    m = v = np.zeros_like(w0)
    w = w0.copy()
    for t in range(1, 4):
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g**2
        w = w - 1e-2 * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
    np.testing.assert_allclose(to_numpy(p), w, atol=1e-6)


def test_ema_matches_formula() -> None:
    lin = nn.Linear(3, 2)
    ema = AveragedModel(lin, multi_avg_fn=get_ema_multi_avg_fn(0.9))
    ref = to_numpy(lin.weight).copy()
    for k in range(4):
        with lucid.no_grad():
            lin.weight.copy_(lin.weight + 1.0)
        ema.update_parameters(lin)
        cur = to_numpy(lin.weight)
        ref = cur if k == 0 else 0.9 * ref + 0.1 * cur  # torch-style: first update copies
    np.testing.assert_allclose(to_numpy(ema.module.weight), ref, atol=1e-5)


@pytest.mark.parametrize("lipschitz", [1.0, 4.0])
def test_spectral_norm_bounds_condition_path(lipschitz: float) -> None:
    """E1.4b depends on it: each linear layer of g has σ_max ≈ 1 after power iteration, so g is L-Lipschitz."""
    gen = MLPGenerator(GeneratorSpec(lipschitz=lipschitz))
    with lucid.no_grad():
        for _ in range(50):                                       # power iterations run in training forwards
            gen.train()
            gen.cond(lucid.rand(8, 2))
    gen.eval()
    for layer in (gen.cond.l1, gen.cond.l2):
        w = to_numpy(layer.weight)
        assert np.linalg.svd(w, compute_uv=False)[0] == pytest.approx(1.0, abs=0.02)
    c = np.random.default_rng(0).uniform(size=(2000, 2))
    out = to_numpy(gen.cond(lucid.tensor(c.astype(np.float32))))
    i, j = np.random.default_rng(1).integers(0, 2000, size=(2, 5000))
    keep = i != j
    ratio = np.linalg.norm(out[i] - out[j], axis=1)[keep] / np.linalg.norm(c[i] - c[j], axis=1)[keep]
    assert ratio.max() <= lipschitz * 1.05


def test_quantize_matches_numpy(rng: np.random.Generator) -> None:
    c = rng.uniform(size=(100, 2))
    got = to_numpy(quantize(lucid.tensor(c.astype(np.float32)), 1 / 16))
    np.testing.assert_allclose(got, (np.floor(c * 16) + 0.5) / 16, atol=1e-6)


def test_generator_gradient_matches_finite_differences(rng: np.random.Generator) -> None:
    gen = MLPGenerator(GeneratorSpec(hidden=16, depth=2, cond_hidden=8))
    eps = lucid.tensor(rng.normal(size=(5, 3, 2)).astype(np.float64), dtype=lucid.float64)
    c = lucid.tensor(rng.uniform(size=(5, 1, 2)).astype(np.float64), dtype=lucid.float64)
    gen = gen.to(dtype=lucid.float64)
    w = gen.head.weight
    loss = (gen(eps, c) ** 2).sum()
    loss.backward()
    analytic = to_numpy(w.grad)[0, 0]
    h = 1e-6
    with lucid.no_grad():
        w[0, 0] += h
        up = float((gen(eps, c) ** 2).sum().item())
        w[0, 0] -= 2 * h
        down = float((gen(eps, c) ** 2).sum().item())
        w[0, 0] += h
    assert analytic == pytest.approx((up - down) / (2 * h), rel=1e-5)
