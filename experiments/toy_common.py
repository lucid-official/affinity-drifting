"""§7 toy 실험 공통부: 2D 데이터 + 가짜 텍스트 임베딩, MLP 생성기, 평가 지표."""

from dataclasses import dataclass

import lucid
import lucid.nn as nn
import numpy as np

TEXT_DIM = 16


@dataclass(frozen=True)
class ToyData:
    """캡션 하나에 이미지 하나인 데이터. `latent`는 정답 조건 변수다."""

    points: np.ndarray  # [M, 2]
    text: np.ndarray  # [M, TEXT_DIM]
    latent: np.ndarray  # [M]


class ToyWorld:
    """`gmm`: 모드 8개 라벨 조건. `ring`: 연속 조건 t, p(x|t)는 서로 반대편인 이봉 분포.

    ring은 조건부 평균이 0이라 회귀로 수렴하면(점 단위 정렬) 바로 드러난다.
    """

    def __init__(self, kind: str, seed: int = 0) -> None:
        if kind not in ("gmm", "ring"):
            raise ValueError(kind)
        self.kind = kind
        rng = np.random.default_rng(seed + 1000)
        self.text_map = np.linalg.qr(rng.normal(size=(TEXT_DIM, TEXT_DIM)))[0]

    def draw_latent(self, count: int, rng: np.random.Generator) -> np.ndarray:
        if self.kind == "gmm":
            return rng.integers(8, size=count)
        return rng.uniform(0, 2 * np.pi, size=count)

    def draw_points(self, latent: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.kind == "gmm":
            angle = latent * 2 * np.pi / 8
            noise = 0.12
        else:
            angle = latent + np.pi * rng.integers(2, size=len(latent))
            noise = 0.08
        points = 2.0 * np.stack([np.cos(angle), np.sin(angle)], axis=-1)
        return (points + rng.normal(scale=noise, size=points.shape)).astype(np.float32)

    def draw_text(self, latent: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """캡션 임베딩. 같은 조건의 다른 표현(paraphrase)을 잡음으로 흉내 낸다."""
        if self.kind == "gmm":
            semantic = np.zeros((len(latent), TEXT_DIM))
            semantic[np.arange(len(latent)), latent] = 1.0
            noise = 0.1
        else:
            semantic = np.zeros((len(latent), TEXT_DIM))
            semantic[:, 0], semantic[:, 1] = np.cos(latent), np.sin(latent)
            noise = 0.03
        semantic += rng.normal(scale=noise, size=semantic.shape)
        return (semantic @ self.text_map.T).astype(np.float32)

    def draw(self, count: int, rng: np.random.Generator) -> ToyData:
        latent = self.draw_latent(count, rng)
        return ToyData(self.draw_points(latent, rng), self.draw_text(latent, rng), latent)


class ToyGenerator(nn.Module):
    def __init__(self, noise_dim: int = 8, hidden: int = 256) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(noise_dim + TEXT_DIM, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, text: lucid.Tensor) -> lucid.Tensor:
        noise = lucid.randn(*text.shape[:-1], self.noise_dim, device=text.device.type)
        return self.net(lucid.cat([noise, text], dim=-1))


def energy_distance(a: np.ndarray, b: np.ndarray) -> float:
    """2E‖a-b‖ - E‖a-a'‖ - E‖b-b'‖. 0이면 두 분포가 같다."""

    def mean_dist(u: np.ndarray, v: np.ndarray) -> float:
        return float(np.linalg.norm(u[:, None] - v[None], axis=-1).mean())

    return 2 * mean_dist(a, b) - mean_dist(a, a) - mean_dist(b, b)


@lucid.no_grad()
def evaluate(generator: ToyGenerator, world: ToyWorld, seed: int = 12345) -> dict[str, float]:
    """훈련에 없던 프롬프트에서 p(x|t)와의 거리를 잰다.

    cond_ed: 프롬프트별 조건부 energy distance 평균. 조건 학습의 핵심 지표.
    marginal_ed: 조건을 무시한 전체 분포의 energy distance.
    spread: 생성 샘플 프롬프트 내 표준편차 / 정답 표준편차. 0에 가까우면 점 단위 수렴이다.
    """
    rng = np.random.default_rng(seed)
    latents = world.draw_latent(48, rng)
    cond, spread_gen, spread_real, all_gen, all_real = [], [], [], [], []
    for value in latents:
        latent = np.full(256, value)
        text = world.draw_text(latent, rng)
        real = world.draw_points(latent, rng)
        fake = generator(lucid.tensor(text)).numpy()
        cond.append(energy_distance(fake, real))
        spread_gen.append(fake.std(0).sum())
        spread_real.append(real.std(0).sum())
        all_gen.append(fake)
        all_real.append(real)
    pooled_gen = np.concatenate(all_gen)[rng.permutation(256 * 48)[:2048]]
    pooled_real = np.concatenate(all_real)[rng.permutation(256 * 48)[:2048]]
    return {
        "cond_ed": float(np.mean(cond)),
        "marginal_ed": energy_distance(pooled_gen, pooled_real),
        "spread": float(np.mean(spread_gen) / np.mean(spread_real)),
    }
