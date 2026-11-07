"""The field: equality with the Lucid release, Lemma 0, Lemma 1, and the arm options."""

import math

import lucid
import lucid.autograd
import numpy as np
import pytest
from lucid.models.generative.drifting import _model as release

from affdrift.core.drift import AffinityDrift, DriftLoss, DriftSpec, RawDriftLoss, Softmax, TargetSet
from affdrift.core.identities import score_ratio_field
from affdrift.core.kernels import XKernel, caption_log_weights
from affdrift.runtime import to_numpy, to_tensor

F64 = lucid.float64


def _t(a: np.ndarray, dtype: lucid.dtype = lucid.float32) -> lucid.Tensor:
    return to_tensor(a, "cpu", dtype)


# ── the base class is the release, unchanged ────────────────────────────────


def test_drift_loss_equals_lucid_release(rng: np.random.Generator) -> None:
    gen = _t(rng.normal(size=(3, 8, 6))).requires_grad_(True)
    pos = _t(rng.normal(size=(3, 5, 6)))
    neg = _t(rng.normal(size=(3, 4, 6)))
    w = _t(rng.uniform(0.5, 2.0, size=(3, 4)))
    temps = (0.02, 0.05, 0.2)

    ours, info = DriftLoss(temps)(gen, pos, neg, w)
    (g_ours,) = lucid.autograd.grad(ours.sum(), gen)
    ref, ref_info = release._drift_loss(gen, pos, neg, w, temps)
    (g_ref,) = lucid.autograd.grad(ref.sum(), gen)

    np.testing.assert_array_equal(to_numpy(ours), to_numpy(ref))
    np.testing.assert_array_equal(to_numpy(g_ours), to_numpy(g_ref))
    assert info.keys() == ref_info.keys()


# ── Lemma 0 (G2 / E1.1) ─────────────────────────────────────────────────────


@pytest.mark.parametrize("kernel", list(XKernel))
def test_lemma0_single_positive_is_l2_regression(rng: np.random.Generator, kernel: XKernel) -> None:
    f = _t(rng.normal(size=(4, 16, 3))).requires_grad_(True)
    y = _t(rng.normal(size=(4, 1, 3)))
    drift = AffinityDrift(DriftSpec(sigma=0.7, kernel=kernel, repulsion=False, drift_norm=False))

    loss, _ = drift(f, y)
    (g_drift,) = lucid.autograd.grad(loss.sum(), f)
    l2 = ((f - y) ** 2).mean(dim=(1, 2))
    (g_l2,) = lucid.autograd.grad(l2.sum(), f)
    assert np.abs(to_numpy(g_drift) - to_numpy(g_l2)).max() < 1e-5


def test_hp_zero_is_single_positive(rng: np.random.Generator) -> None:
    """h_p → 0 over the whole bank is the single-positive field (same code path)."""
    psi = rng.uniform(size=(10, 1))
    y = rng.normal(size=(10, 2))
    log_w = caption_log_weights(psi, psi, 0.0)
    np.testing.assert_array_equal(np.exp(log_w), np.eye(10))

    x = _t(rng.normal(size=(10, 32, 2)))
    drift = AffinityDrift(DriftSpec(sigma=0.5, repulsion=False, drift_norm=False))
    bank = TargetSet(_t(y[None]), _t(log_w[:, None, :]))
    v = drift.field(x, bank).v
    expected = _t(y[:, None, :]) - x
    assert np.abs(to_numpy(v - expected)).max() < 1e-6


# ── Lemma 1 (E0.1) ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("sigma", [0.1, 0.5, 2.0])
def test_lemma1_field_is_scaled_score_ratio(rng: np.random.Generator, sigma: float) -> None:
    b, c, p, s = 3, 24, 40, 2
    x = _t(rng.normal(size=(b, c, s)), F64)
    log_w = caption_log_weights(rng.uniform(size=(b, 1)), rng.uniform(size=(p, 1)), 0.2)
    pos = TargetSet(_t(rng.normal(size=(1, p, s)), F64), _t(log_w[:, None, :], F64))
    gen = TargetSet.own_samples(x)
    drift = AffinityDrift(DriftSpec(sigma=sigma, drift_norm=False))

    v = drift.field(x, pos, gen).v
    ref = score_ratio_field(x, pos, gen, sigma)
    assert np.abs(to_numpy(v) - to_numpy(ref)).max() < 1e-6


# ── arm options ─────────────────────────────────────────────────────────────


def test_k1_target_is_independent_of_x(rng: np.random.Generator) -> None:
    y = _t(rng.normal(size=(1, 6, 2)))
    drift = AffinityDrift(DriftSpec(sigma=0.3, kernel=XKernel.NONE, repulsion=False, drift_norm=False))
    x = _t(rng.normal(size=(1, 50, 2)))
    target = to_numpy(drift.field(x, TargetSet.uniform(y)).v + x)
    np.testing.assert_allclose(target, np.broadcast_to(target[:, :1], target.shape), atol=1e-6)


def test_joint_softmax_is_mass_weighted_separate_field(rng: np.random.Generator) -> None:
    x = _t(rng.normal(size=(2, 12, 2)), F64)
    pos = TargetSet.uniform(_t(rng.normal(size=(2, 7, 2)), F64))
    gen = TargetSet.own_samples(x)
    spec = DriftSpec(sigma=0.8, drift_norm=False)
    sep = AffinityDrift(spec)
    joint = AffinityDrift(DriftSpec(sigma=0.8, drift_norm=False, softmax=Softmax.JOINT))

    logits = lucid.cat([sep.logits(x, gen), sep.logits(x, pos)], dim=-1)
    a = lucid.exp(logits - lucid.logsumexp(logits, dim=-1, keepdim=True))
    z_gen, z_pos = a[..., :12].sum(dim=-1, keepdim=True), a[..., 12:].sum(dim=-1, keepdim=True)
    expected = z_gen * z_pos * sep.field(x, pos, gen).v
    assert np.abs(to_numpy(joint.field(x, pos, gen).v - expected)).max() < 1e-10


def test_own_samples_excludes_self() -> None:
    x = lucid.randn(2, 5, 3)
    w = to_numpy(lucid.exp(TargetSet.own_samples(x).log_w))[0]
    np.testing.assert_allclose(np.diag(w), 0.0)
    np.testing.assert_allclose(w.sum(axis=-1), 1.0, rtol=1e-6)


def test_drift_norm_gives_unit_energy(rng: np.random.Generator) -> None:
    x = _t(rng.normal(size=(4, 16, 3)))
    pos = TargetSet.uniform(_t(rng.normal(size=(4, 9, 3))))
    terms = AffinityDrift(DriftSpec(sigma=1.0)).field(x, pos)
    for t in (terms.v_plus, terms.v_minus):
        assert math.isclose(float((t * t).mean().item()), 1.0, rel_tol=1e-4)


def test_k3_requires_generator() -> None:
    with pytest.raises(ValueError, match="generator"):
        AffinityDrift(DriftSpec(sigma=1.0, pos_noise_std=0.1))


def test_affd_rejects_release_cfg_weights(rng: np.random.Generator) -> None:
    drift = AffinityDrift(DriftSpec(sigma=1.0))
    x, y = _t(rng.normal(size=(1, 4, 2))), _t(rng.normal(size=(1, 2, 2)))
    with pytest.raises(TypeError):
        drift(x, y, y, _t(np.ones((1, 2))))


# ── the baseline arm ────────────────────────────────────────────────────────


def test_raw_release_field_is_energy_times_normalised(rng: np.random.Generator) -> None:
    """RawDriftLoss with one τ is the release force times √energy, and has the same zeros."""
    gen = _t(rng.normal(size=(2, 10, 3)))
    pos = _t(rng.normal(size=(2, 4, 3)))
    norm_v = DriftLoss((0.05,)).drift(gen, pos)
    raw = RawDriftLoss((0.05,))
    _, _, info = raw.goal(gen, pos)
    raw_v = raw.drift(gen, pos)
    scale = float(np.sqrt(max(float(info["loss_0.05"].item()), 1e-8)))
    np.testing.assert_allclose(to_numpy(raw_v), to_numpy(norm_v) * scale, rtol=1e-5, atol=1e-7)


def test_affd_drift_is_its_field(rng: np.random.Generator) -> None:
    x = _t(rng.normal(size=(2, 8, 2)))
    pos = TargetSet.uniform(_t(rng.normal(size=(2, 5, 2))))
    drift = AffinityDrift(DriftSpec(sigma=0.5, drift_norm=False))
    np.testing.assert_allclose(to_numpy(drift.drift(x, pos)), to_numpy(drift.field(x, pos).v), atol=1e-6)


# ── multi-scale field ───────────────────────────────────────────────────────


def test_multiscale_with_one_scale_is_affd(rng: np.random.Generator) -> None:
    from affdrift.core.drift import MultiScaleAffinityDrift

    x = _t(rng.normal(size=(2, 10, 2)))
    pos = TargetSet.uniform(_t(rng.normal(size=(2, 6, 2))))
    spec = DriftSpec(sigma=0.4)
    np.testing.assert_allclose(to_numpy(MultiScaleAffinityDrift(spec, [0.4]).field(x, pos).v),
                               to_numpy(AffinityDrift(spec).field(x, pos).v), atol=1e-6)


def test_multiscale_raw_field_is_sum_of_scaled_score_ratios(rng: np.random.Generator) -> None:
    """Lemma 1 per scale: V = Σ_k σ_k² ∇log(p̂_k / q̂_k) when nothing is normalised."""
    from affdrift.core.drift import MultiScaleAffinityDrift

    x = _t(rng.normal(size=(2, 12, 2)), F64)
    pos = TargetSet.uniform(_t(rng.normal(size=(2, 20, 2)), F64))
    gen = TargetSet.own_samples(x)
    sigmas = (0.1, 0.4, 1.6)
    v = MultiScaleAffinityDrift(DriftSpec(sigma=1.0, drift_norm=False), sigmas).field(x, pos, gen).v
    ref = sum(to_numpy(score_ratio_field(x, pos, gen, s)) for s in sigmas)
    assert np.abs(to_numpy(v) - ref).max() < 1e-9
