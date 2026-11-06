"""Drifting field 코어.

논문: Deng et al., *Generative Modeling via Drifting* (arXiv 2602.04770)
- Alg.2        `drift_from_logits`, `compute_v`
- A.6          `normalized_drift` (특징 정규화 S, 드리프트 정규화 λ, 다중 temperature)
- A.7          `cfg_log_weights` (식 15–16의 혼합 가중치 w)
- 식 (26)      `feature_drift_loss`

아이디어 문서 §3.2의 joint kernel은 논문 kernel의 logit에 텍스트 항을 더하는 것으로 구현한다:

    log k((x,t),(y,s)) = -‖φ(x)-φ(y)‖/τ_x + log_k_text(t,s)

drift는 이미지 좌표에만 작용하므로 Alg.2의 나머지 부분은 그대로다. 텍스트 항은 양/음 샘플에
똑같이 들어가므로 anti-symmetry(V_{p,q} = -V_{q,p})가 보존된다. tests/test_drift.py에서 확인한다.

모든 함수는 `[..., N, C]` 배치 차원을 지원한다. 학습에서는 `[G, L, N, C]`를 쓴다.
G는 그룹(논문의 클래스 Nc), L은 위치(A.5), N은 샘플, C는 특징 차원이다.
"""

from dataclasses import dataclass
from math import sqrt

import lucid
import lucid.nn.functional as F
from lucid import Tensor

# 마스킹용 log-kernel. Alg.2의 `dist_neg += eye(N) * 1e6`과 같은 역할이다.
LOG_ZERO = -1e9

DEFAULT_TEMPERATURES = (0.02, 0.05, 0.2)  # 논문 Table 8


def pairwise_distance(x: Tensor, y: Tensor) -> Tensor:
    """`[..., N, C]`, `[..., M, C]` -> `[..., N, M]` ℓ2 거리.

    차원별 뺄셈 대신 내적 전개를 쓴다. `[G, L, N, M, C]` 중간 텐서를 만들지 않아 메모리가 훨씬 적다.
    """
    xx = (x * x).sum(-1, keepdim=True)
    yy = (y * y).sum(-1, keepdim=True).swapaxes(-1, -2)
    squared = xx + yy - 2.0 * (x @ y.swapaxes(-1, -2))
    return lucid.sqrt(lucid.clamp(squared, min=0.0) + 1e-12)


def drift_from_logits(
    logit_pos: Tensor,
    logit_neg: Tensor,
    y_pos: Tensor,
    y_neg: Tensor,
    normalize: str = "both",
) -> Tensor:
    """Alg.2의 logit 이후 부분. `normalize`는 "both"(논문 기본) 또는 "y"(y축 softmax만)."""
    num_pos = int(logit_pos.shape[-1])
    logit = lucid.cat([logit_pos, logit_neg], dim=-1)
    log_row = F.log_softmax(logit, dim=-1)
    if normalize == "both":
        # sqrt(A_row * A_col)을 log 공간에서 계산한다. 마스킹된 항의 A가 정확히 0이어도
        # sqrt의 기울기가 발산하지 않아, V를 통과해 미분하는 실험(§7 b)에서도 안전하다.
        affinity = lucid.exp(0.5 * (log_row + F.log_softmax(logit, dim=-2)))
    elif normalize == "y":
        affinity = lucid.exp(log_row)
    else:
        raise ValueError(f"unknown normalize mode: {normalize!r}")
    a_pos = affinity[..., :num_pos]
    a_neg = affinity[..., num_pos:]
    w_pos = a_pos * a_neg.sum(-1, keepdim=True)
    w_neg = a_neg * a_pos.sum(-1, keepdim=True)
    return w_pos @ y_pos - w_neg @ y_neg


def _self_mask(num_x: int, num_neg: int, *, device: str) -> Tensor:
    """y_neg의 앞 N개가 x 자신일 때 대각 성분을 지우는 log-kernel."""
    eye = lucid.eye(num_x, device=device)
    if num_neg > num_x:
        eye = lucid.cat([eye, lucid.zeros(num_x, num_neg - num_x, device=device)], dim=-1)
    return eye * LOG_ZERO


def _add(a: Tensor, b: Tensor | None) -> Tensor:
    return a if b is None else a + b


def compute_v(
    x: Tensor,
    y_pos: Tensor,
    y_neg: Tensor,
    temperature: float,
    *,
    mask_self: bool = False,
    log_k_pos: Tensor | None = None,
    log_k_neg: Tensor | None = None,
    normalize: str = "both",
) -> Tensor:
    """논문 Alg.2 `compute_V`. `log_k_*`를 주면 joint kernel이 된다."""
    logit_pos = _add(-pairwise_distance(x, y_pos) / temperature, log_k_pos)
    logit_neg = _add(-pairwise_distance(x, y_neg) / temperature, log_k_neg)
    if mask_self:
        logit_neg = logit_neg + _self_mask(
            int(x.shape[-2]), int(y_neg.shape[-2]), device=x.device.type
        )
    return drift_from_logits(logit_pos, logit_neg, y_pos, y_neg, normalize)


def text_log_kernel(e_a: Tensor, e_b: Tensor, tau_t: float) -> Tensor:
    """아이디어 문서 §3.2의 텍스트 항 `-‖e(t)-e(s)‖/τ_t`.

    `e_*`는 `TextKernel`이 whitening하고 스케일을 맞춘 키여야 τ_t의 의미가 데이터에 무관해진다.
    """
    return -pairwise_distance(e_a, e_b) / tau_t


def cfg_log_weights(alpha: Tensor, log_k_neg: Tensor | None, num_x: int, num_unc: int) -> Tensor:
    """A.7의 비조건 샘플 가중치 log w.

    논문: α = ((N_neg-1) + N_unc·w) / (N_neg-1)  =>  w = (α-1)(N_neg-1)/N_unc.

    joint kernel에서는 "프롬프트 t_i의 음성 샘플 수"가 정수가 아니므로 유효 개수로 일반화한다.

        n_i = Σ_{j≠i} k_txt(t_i, t_j),   w_i = (α_i-1)·n_i / N_unc

    k_txt가 같은 클래스면 1이면 n_i = N_neg-1이 되어 논문 식과 같아진다. α=1이면 w=0(항 제거)이다.

    `alpha`는 `[..., N]`, 반환은 `[..., 1, N, U]`로 `[G, L, N, U]`에 브로드캐스트된다.
    """
    device = alpha.device.type
    if log_k_neg is None:
        n_eff = lucid.full_like(alpha, float(num_x - 1))
    else:
        mask = _self_mask(num_x, int(log_k_neg.shape[-1]), device=device)
        n_eff = lucid.exp(log_k_neg + mask).sum(-1)
        n_eff = n_eff.reshape(alpha.shape)
    weight = (alpha - 1.0) * n_eff / float(num_unc)
    positive = weight > 0
    log_weight = lucid.where(positive, lucid.log(lucid.clamp(weight, min=1e-30)), LOG_ZERO)
    # [..., N] -> [..., 1, N, U]
    log_weight = log_weight.unsqueeze(-1).repeat_interleave(num_unc, dim=-1)
    return log_weight.unsqueeze(-3)


def sample_alpha(count: int, *, alpha_max: float = 4.0, power: float = 3.0) -> Tensor:
    """p(α) ∝ α^{-power} on [1, alpha_max] (논문 Table 8). 역CDF 샘플링."""
    u = lucid.rand(count)
    exponent = 1.0 - power
    return (1.0 - u * (1.0 - alpha_max**exponent)) ** (1.0 / exponent)


@dataclass(frozen=True)
class DriftKernel:
    """한 배치의 텍스트·CFG log-kernel. 모두 `[G, L, N, *]`에 브로드캐스트 가능해야 한다.

    전부 None이면 그룹 안의 모든 샘플이 같은 분포에서 왔다고 보는 논문 Alg.1과 같다.
    """

    log_k_pos: Tensor | None = None  # [..., N, P]
    log_k_neg: Tensor | None = None  # [..., N, N]
    log_w_unc: Tensor | None = None  # [..., N, U]


@dataclass(frozen=True)
class DriftStats:
    """`raw_drift`는 τ별 E[‖V‖²/C]의 평균이다.

    A.6의 λ 정규화 후에는 loss가 설계상 거의 상수이므로 수렴 확인에는 이 값을 봐야 한다.
    `.item()`을 매번 부르면 MLX 지연 평가가 끊기므로, 로그를 남길 때만 값을 꺼내도록 텐서로 둔다.
    """

    scale: Tensor
    raw_drift: Tensor


def _weighted_mean(values: Tensor, log_weight: Tensor | None) -> tuple[Tensor, Tensor]:
    weight = lucid.ones_like(values)
    if log_weight is not None:
        weight = weight * lucid.exp(log_weight)
    return (values * weight).sum(), weight.sum()


def normalized_drift(
    x: Tensor,
    y_pos: Tensor,
    y_unc: Tensor | None,
    kernel: DriftKernel,
    temperatures: tuple[float, ...] = DEFAULT_TEMPERATURES,
    *,
    normalize: str = "both",
    scale: Tensor | None = None,
    normalize_drift: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """A.6 정규화를 적용한 `Ṽ = Σ_τ V_τ/λ_τ`를 반환한다.

    Returns:
        (Ṽ, S, raw): Ṽ는 x와 같은 shape, S는 스칼라 스케일, raw는 τ 평균 E[‖V‖²/C].

    `scale`을 주면 배치 통계 대신 그 값을 쓴다. 후보끼리 ‖V‖²를 비교할 때(§6) 필요하다.
    `normalize_drift=False`면 λ 나눔을 건너뛴다. 이때 Ṽ는 정규화 공간의 원래 drift 합이다.
    """
    num_x, dim = int(x.shape[-2]), int(x.shape[-1])
    device = x.device.type
    y_neg = x if y_unc is None else lucid.cat([x, y_unc], dim=-2)
    dist_pos = pairwise_distance(x, y_pos)
    dist_neg = pairwise_distance(x, y_neg)

    # 음성 쪽 log-kernel: 텍스트 항 + 비조건 샘플의 log w + 자기 자신 제거.
    num_neg = int(y_neg.shape[-2])
    log_neg = _self_mask(num_x, num_neg, device=device)
    if kernel.log_k_neg is not None or kernel.log_w_unc is not None:
        extra = kernel.log_k_neg
        if y_unc is not None:
            unc = kernel.log_w_unc
            if unc is None:
                unc = lucid.zeros(*dist_neg.shape[:-1], num_neg - num_x, device=device)
            base = (
                lucid.zeros(*dist_neg.shape[:-1], num_x, device=device) if extra is None else extra
            )
            base, unc = _broadcast_pair(base, unc)
            extra = lucid.cat([base, unc], dim=-1)
        log_neg = log_neg + extra

    if scale is None:
        # 식 (21). CFG의 w와 텍스트 kernel도 같은 가중치로 반영한다(A.7 끝부분).
        sum_p, weight_p = _weighted_mean(dist_pos, kernel.log_k_pos)
        sum_n, weight_n = _weighted_mean(dist_neg, log_neg)
        scale = (sum_p + sum_n) / (weight_p + weight_n) / sqrt(dim)
        scale = lucid.clamp(scale.detach(), min=1e-8)

    y_pos_n = y_pos / scale
    y_neg_n = y_neg / scale
    total: Tensor | None = None
    raw_sum: Tensor | None = None
    for tau in temperatures:
        temp = scale * tau * sqrt(dim)  # 식 (22): τ̃ = τ√C, dist/S 를 쓰는 것과 동일.
        logit_pos = _add(-dist_pos / temp, kernel.log_k_pos)
        logit_neg = -dist_neg / temp + log_neg
        field = drift_from_logits(logit_pos, logit_neg, y_pos_n, y_neg_n, normalize)
        raw = (field * field).sum(-1).mean() / dim
        if normalize_drift:
            field = field / (lucid.sqrt(raw.detach()) + 1e-12)  # 식 (25)
        total = field if total is None else total + field
        raw_sum = raw if raw_sum is None else raw_sum + raw
    assert total is not None and raw_sum is not None, "temperatures must not be empty"
    return total, scale, raw_sum / len(temperatures)


def _broadcast_pair(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """마지막 축을 제외하고 두 텐서를 같은 shape으로 맞춘다. cat 앞에서 쓴다."""
    head = lucid.broadcast_shapes(tuple(a.shape[:-1]), tuple(b.shape[:-1]))
    return (
        a.expand(*head, int(a.shape[-1])),
        b.expand(*head, int(b.shape[-1])),
    )


def feature_drift_loss(
    phi_x: Tensor,
    phi_pos: Tensor,
    phi_unc: Tensor | None = None,
    kernel: DriftKernel | None = None,
    temperatures: tuple[float, ...] = DEFAULT_TEMPERATURES,
    *,
    normalize: str = "both",
) -> tuple[Tensor, DriftStats]:
    """식 (26): `MSE(φ̃(x), stopgrad(φ̃(x) + Ṽ))`. 기울기는 예측 항 φ(x)로만 흐른다."""
    kernel = kernel or DriftKernel()
    with lucid.no_grad():
        field, scale, raw = normalized_drift(
            phi_x.detach(),
            phi_pos.detach(),
            None if phi_unc is None else phi_unc.detach(),
            kernel,
            temperatures,
            normalize=normalize,
        )
        target = phi_x.detach() / scale + field
    loss = F.mse_loss(phi_x / scale, target)
    return loss, DriftStats(scale=scale, raw_drift=raw)
