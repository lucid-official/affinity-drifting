"""논문과 아이디어 문서가 주장하는 코어 성질을 수치로 확인한다."""

from math import sqrt

import lucid
import numpy as np
import pytest

from clipdrift.drift import (
    DriftKernel,
    cfg_log_weights,
    compute_v,
    drift_from_logits,
    feature_drift_loss,
    normalized_drift,
    pairwise_distance,
    sample_alpha,
    text_log_kernel,
)


@pytest.fixture(autouse=True)
def _seed():
    lucid.manual_seed(0)


def close(a, b, tol=1e-5):
    return float((a - b).abs().max().item()) < tol


def test_pairwise_distance_matches_cdist():
    x, y = lucid.randn(5, 7), lucid.randn(9, 7)
    assert close(pairwise_distance(x, y), lucid.cdist(x, y), 1e-4)


def test_alg2_matches_hand_computed_reference():
    """Alg.2를 행렬 연산 없이 원소 단위로 다시 계산해 비교한다."""
    x, y_pos, y_neg = lucid.randn(3, 2), lucid.randn(4, 2), lucid.randn(5, 2)
    temp = 0.7
    v = compute_v(x, y_pos, y_neg, temp)

    d_pos = lucid.cdist(x, y_pos).numpy()
    d_neg = lucid.cdist(x, y_neg).numpy()
    logit = np.concatenate([-d_pos / temp, -d_neg / temp], axis=1)
    e = np.exp(logit)
    row = e / e.sum(axis=1, keepdims=True)
    col = e / e.sum(axis=0, keepdims=True)
    a = np.sqrt(row * col)
    a_pos, a_neg = a[:, :4], a[:, 4:]
    w_pos = a_pos * a_neg.sum(axis=1, keepdims=True)
    w_neg = a_neg * a_pos.sum(axis=1, keepdims=True)
    expected = w_pos @ y_pos.numpy() - w_neg @ y_neg.numpy()
    assert np.abs(v.numpy() - expected).max() < 1e-5


def test_antisymmetry():
    """V_{p,q} = -V_{q,p} (식 3)."""
    x, a, b = lucid.randn(6, 3), lucid.randn(8, 3), lucid.randn(8, 3)
    assert close(compute_v(x, a, b, 0.5), -compute_v(x, b, a, 0.5))


def test_zero_drift_when_p_equals_q():
    """양/음 샘플이 같으면 V = 0 (명제 3.1)."""
    x, y = lucid.randn(6, 3), lucid.randn(8, 3)
    assert close(compute_v(x, y, y, 0.5), lucid.zeros_like(x))


def test_joint_kernel_preserves_antisymmetry_and_zero_drift():
    """아이디어 문서 §3.2: 텍스트 항을 양/음에 똑같이 넣으면 두 성질이 유지된다."""
    x, a, b = lucid.randn(6, 3), lucid.randn(8, 3), lucid.randn(8, 3)
    t_x, t_a, t_b = lucid.randn(6, 4), lucid.randn(8, 4), lucid.randn(8, 4)
    k_a, k_b = text_log_kernel(t_x, t_a, 0.5), text_log_kernel(t_x, t_b, 0.5)
    forward = compute_v(x, a, b, 0.5, log_k_pos=k_a, log_k_neg=k_b)
    backward = compute_v(x, b, a, 0.5, log_k_pos=k_b, log_k_neg=k_a)
    assert close(forward, -backward)
    same = compute_v(x, a, a, 0.5, log_k_pos=k_a, log_k_neg=k_a)
    assert close(same, lucid.zeros_like(x))


def test_hard_text_kernel_equals_per_class_alg2():
    """아이디어 문서 §3.2: 클래스 조건은 "같은 클래스면 k_txt=1"인 특수한 경우다."""
    a_x, b_x = lucid.randn(4, 2), lucid.randn(4, 2)
    a_y, b_y = lucid.randn(5, 2), lucid.randn(5, 2)
    a_n, b_n = lucid.randn(6, 2), lucid.randn(6, 2)
    per_class = lucid.cat(
        [compute_v(a_x, a_y, a_n, 0.5), compute_v(b_x, b_y, b_n, 0.5)], dim=0
    )

    def block(rows, cols):
        top = lucid.cat([lucid.zeros(rows, cols), lucid.full((rows, cols), -1e9)], dim=1)
        bottom = lucid.cat([lucid.full((rows, cols), -1e9), lucid.zeros(rows, cols)], dim=1)
        return lucid.cat([top, bottom], dim=0)

    joint = compute_v(
        lucid.cat([a_x, b_x], dim=0),
        lucid.cat([a_y, b_y], dim=0),
        lucid.cat([a_n, b_n], dim=0),
        0.5,
        log_k_pos=block(4, 5),
        log_k_neg=block(4, 6),
    )
    assert close(joint, per_class)


def test_batched_equals_loop():
    """`[G, L, N, C]` 배치 계산이 하나씩 계산한 결과와 같아야 한다."""
    x, y_pos, y_neg = lucid.randn(2, 3, 4, 5), lucid.randn(2, 3, 6, 5), lucid.randn(2, 3, 7, 5)
    batched = compute_v(x, y_pos, y_neg, 0.5)
    for g in range(2):
        for l in range(3):
            assert close(batched[g, l], compute_v(x[g, l], y_pos[g, l], y_neg[g, l], 0.5))


def test_mask_self_ignores_own_sample():
    """x를 크게 옮겨도 자기 자신이 음성으로 잡혀 V가 달라지면 안 된다."""
    x = lucid.randn(4, 2)
    y_pos = lucid.randn(4, 2)
    masked = compute_v(x, y_pos, x, 0.5, mask_self=True)
    dist = pairwise_distance(x, x) + lucid.eye(4) * 1e6
    logit_pos = -pairwise_distance(x, y_pos) / 0.5
    expected = drift_from_logits(logit_pos, -dist / 0.5, y_pos, x)
    assert close(masked, expected)


def test_scale_normalization_makes_field_invariant():
    """A.6: 특징 전체를 c배 해도 정규화된 drift는 변하지 않는다."""
    x, y = lucid.randn(1, 2, 16, 8), lucid.randn(1, 2, 16, 8)
    field_a, scale_a, raw_a = normalized_drift(x, y, None, DriftKernel())
    field_b, scale_b, raw_b = normalized_drift(x * 37.0, y * 37.0, None, DriftKernel())
    assert close(field_a, field_b, 1e-3)
    assert abs(scale_b.item() / scale_a.item() - 37.0) < 1e-2
    assert abs(raw_a.item() - raw_b.item()) < 1e-3


def test_feature_scale_gives_mean_distance_sqrt_c():
    """식 (20): 정규화 후 평균 거리는 √C."""
    x, y = lucid.randn(32, 64), lucid.randn(32, 64)
    _, scale, _ = normalized_drift(x, y, None, DriftKernel())
    d_pos = lucid.cdist(x / scale, y / scale)
    d_neg = lucid.cdist(x / scale, x / scale)
    total = d_pos.sum() + d_neg.sum()
    mean = (total / (32 * 32 + 32 * 31)).item()
    assert abs(mean - sqrt(64)) / sqrt(64) < 1e-3


def test_drift_normalization_gives_unit_field():
    """식 (24): τ 하나일 때 E[‖Ṽ‖²/C] ≈ 1."""
    x, y = lucid.randn(32, 64), lucid.randn(32, 64)
    field, _, _ = normalized_drift(x, y, None, DriftKernel(), (0.05,))
    assert abs((field * field).sum(-1).mean().item() / 64 - 1.0) < 1e-3


def test_loss_gradient_flows_only_through_prediction():
    x = lucid.randn(1, 1, 8, 4, requires_grad=True)
    y = lucid.randn(1, 1, 8, 4)
    loss, stats = feature_drift_loss(x, y)
    loss.backward()
    assert stats.raw_drift.item() > 0
    with lucid.no_grad():
        field, scale, _ = normalized_drift(x.detach(), y, None, DriftKernel())
        # d/dx mean((x/S - stopgrad(x/S + V))²) = -2·V/(S·numel)
        expected = -2.0 * field / scale / float(8 * 4)
    assert close(x.grad, expected, 1e-4)


def test_cfg_weight_matches_paper_formula_for_hard_classes():
    """k_txt가 전부 1이면 w = (α-1)(N_neg-1)/N_unc (A.7)."""
    alpha = lucid.tensor([[1.0, 2.0, 3.0, 4.0]])
    log_w = cfg_log_weights(alpha, None, num_x=4, num_unc=2)
    assert log_w.shape == (1, 1, 4, 2)
    expected = [(a - 1.0) * 3 / 2 for a in (2.0, 3.0, 4.0)]
    for i, w in enumerate(expected, start=1):
        assert abs(lucid.exp(log_w[0, 0, i, 0]).item() - w) < 1e-4
    assert log_w[0, 0, 0, 0].item() < -1e8  # α=1이면 항이 사라진다.


def test_cfg_weight_uses_effective_negative_count():
    """모든 프롬프트가 같으면 soft kernel도 논문 식과 같아져야 한다."""
    e = lucid.ones(4, 3)
    log_k = text_log_kernel(e, e, 0.5)
    alpha = lucid.full((4,), 3.0)
    soft = cfg_log_weights(alpha, log_k, num_x=4, num_unc=2)
    hard = cfg_log_weights(alpha, None, num_x=4, num_unc=2)
    assert close(soft, hard, 1e-3)


def test_cfg_mixture_is_exact_with_exchangeable_samples():
    """식 (15): α로 만든 음성 혼합이 양성 분포와 같으면 기대 drift가 0이다.

    x 하나에 대해 양성 집합과 (생성 + w배 비조건) 음성 집합을 같은 분포에서 뽑아 평균을 낸다.
    """
    x = lucid.zeros(8, 2)
    total = lucid.zeros(8, 2)
    trials = 400
    for _ in range(trials):
        y_pos = lucid.randn(32, 2)
        neg = lucid.randn(8, 2)
        unc = lucid.randn(8, 2)
        alpha = lucid.full((8,), 2.0)
        log_w = cfg_log_weights(alpha, None, num_x=8, num_unc=8).squeeze(0)
        logit_neg = lucid.cat(
            [-pairwise_distance(x, neg) + (lucid.eye(8) * -1e9), -pairwise_distance(x, unc) + log_w],
            dim=-1,
        )
        y_neg = lucid.cat([neg, unc], dim=0)
        total = total + drift_from_logits(-pairwise_distance(x, y_pos), logit_neg, y_pos, y_neg)
    assert (total / trials).abs().max().item() < 0.05


def test_sample_alpha_range_and_skew():
    alpha = sample_alpha(20000, alpha_max=4.0, power=3.0)
    assert alpha.min().item() >= 1.0 and alpha.max().item() <= 4.0 + 1e-5
    # p(α) ∝ α^-3 on [1, 4]의 중앙값 = (1 - 0.5·(1 - 1/16))^(-1/2) = 1.3720..
    median = float(np.median(alpha.numpy()))
    assert abs(median - 1.3720) < 0.02


def test_runs_on_metal():
    x = lucid.randn(1, 2, 8, 4, device="metal", requires_grad=True)
    y = lucid.randn(1, 2, 8, 4, device="metal")
    u = lucid.randn(1, 2, 3, 4, device="metal")
    alpha = lucid.full((8,), 2.0, device="metal")
    kernel = DriftKernel(log_w_unc=cfg_log_weights(alpha, None, 8, 3))
    loss, _ = feature_drift_loss(x, y, u, kernel)
    loss.eval()
    loss.backward()
    x.grad.eval()
    assert x.grad.device.type == "metal"
    assert float(x.grad.abs().sum().item()) > 0
