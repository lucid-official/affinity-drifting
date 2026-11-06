"""특징 공간 φ (논문 §3.4, A.3, A.5).

특징 추출기는 `images -> {이름: [B, L, C]}`인 함수다. L은 위치 수, C는 차원이다.
이름 하나마다 drifting loss 하나를 계산하고 전부 합산한다(식 14).
"""

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid import Tensor


def _std(mean_of_square: Tensor, mean: Tensor) -> Tensor:
    return lucid.sqrt(lucid.clamp(mean_of_square - mean * mean, min=0.0) + 1e-8)


def map_vectors(prefix: str, feature_map: Tensor, out: dict[str, Tensor]) -> None:
    """특징 맵 `[B, C, H, W]` -> A.5의 (a)~(d) 벡터.

    (a) 위치별 벡터, (b) 전역 평균·표준편차, (c) 2×2 평균·표준편차, (d) 4×4 평균·표준편차.
    """
    batch, channels, height, width = (int(v) for v in feature_map.shape)
    out[f"{prefix}/loc"] = feature_map.reshape(batch, channels, -1).swapaxes(1, 2)
    square = feature_map * feature_map
    global_mean = feature_map.mean(dim=(2, 3))
    out[f"{prefix}/mean"] = global_mean.unsqueeze(1)
    out[f"{prefix}/std"] = _std(square.mean(dim=(2, 3)), global_mean).unsqueeze(1)
    for patch in (2, 4):
        if height < patch or width < patch:
            continue
        mean = F.avg_pool2d(feature_map, patch, patch)
        std = _std(F.avg_pool2d(square, patch, patch), mean)
        out[f"{prefix}/mean{patch}"] = mean.reshape(batch, channels, -1).swapaxes(1, 2)
        out[f"{prefix}/std{patch}"] = std.reshape(batch, channels, -1).swapaxes(1, 2)


def pixel_features(images: Tensor) -> dict[str, Tensor]:
    """φ 없는 원본 공간 drift (A.5 끝부분).

    논문 A.5의 입력 x² 평균 항은 일부러 뺐다. 이 항의 기울기는 2x에 비례해 픽셀 부호를 증폭하고, 목표값은
    x=+0.9와 x=-0.9에서 같아서 둘을 구분하지 못한다. shapes 데이터에서는 이 항 때문에 프롬프트와 무관하게
    출력 전체가 한 극으로 붕괴했다. 논문은 평균이 0 근처인 잠재 공간에서 쓴다.
    """
    return {"pixel/raw": images.reshape(int(images.shape[0]), 1, -1)}


def subsample_locations(
    features: tuple[dict[str, Tensor], ...],
    max_locations: int | None,
) -> tuple[dict[str, Tensor], ...]:
    """이름별로 위치를 무작위로 줄인다. 같은 위치를 모든 집합에 적용한다.

    논문은 전 위치를 쓰지만 18GB 메모리와 작은 배치에 맞춰 줄일 수 있게 했다.
    """
    if max_locations is None:
        return features
    first = features[0]
    selected = tuple({} for _ in features)
    for name, value in first.items():
        count = int(value.shape[1])
        if count <= max_locations:
            for out, source in zip(selected, features):
                out[name] = source[name]
            continue
        index = lucid.randperm(count, device=value.device.type)[:max_locations]
        for out, source in zip(selected, features):
            out[name] = source[name][:, index]
    return selected


def freeze(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        parameter.requires_grad = False
    return module.eval()
